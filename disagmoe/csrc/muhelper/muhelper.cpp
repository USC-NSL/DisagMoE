#include <condition_variable>
#include <cstdlib>
#include <string>
#include <mutex>
#include <queue>
#include <ctime>
#include <utility>

#include "datatypes.hpp"
#include "muhelper.h"
#include "comm.h"
#include "utils.hpp"
#include "logging.h"
#include "constants.h"

#include "zmq.hpp"
#include "zmq_addon.hpp"

#include <cereal/archives/binary.hpp>

// MuHelper

MuHelper::MuHelper(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    layer_ids(layer_ids), device_id(device_id), channels(channels), end_flag(false) {
        LOG(INFO) << "init muhelper@" << device_id << LEND;
    }

void MuHelper::start() {
    LOG(INFO) << "muhelper@" << device_id << " start" << LEND;
    this->thread = std::thread(
        [&](MuHelper* helper) {
            helper->init_cuda_device();
            helper->run(); 
        }, 
        this
    );
}

int MuHelper::get_device_id() {
    return device_id;
}

void MuHelper::terminate() {
    this->end_flag = true;
    this->thread.join();
}

void MuHelper::init_cuda_device() {
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->device_id));
    #endif
}

// MuDispatcher

MuDispatcher::MuDispatcher(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    MuHelper(layer_ids, device_id, channels), 
    peer_ctx(channels.size()),
    peer_mq(channels.size()) {
    sprintf(this->device_id_str, "%d", this->device_id);
    for (int i = 0; i < channels.size(); i ++) {
        peer_ctx[i] = zmq::context_t(1);
        peer_mq[i] = zmq::socket_t(peer_ctx[i], zmq::socket_type::push);
    }
}

void MuDispatcher::_send_batch(int cid, uintptr_t buf, const Metadata& meta) {
    auto data = cerealize(std::make_shared<Metadata>(meta));
    LOG(DEBUG) << "sending batch to channel " << cid << LEND;
    this->peer_mq[cid].send(zmq::str_buffer(this->device_id_str), zmq::send_flags::sndmore);
    this->peer_mq[cid].send(zmq::buffer(data.c_str(), data.size()));
    this->channels[cid]->send(buf, meta);
}

void MuDispatcher::run() {
    for (int i = 0; i < this->channels.size(); i ++)
        this->peer_mq[i].connect(get_zmq_addr(this->channels[i]->get_peer_id()));

    while (!this->end_flag) {
        std::unique_lock<std::mutex> lock(this->mtx);
        this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
        auto batch = this->send_queue.front();
        this->send_queue.pop();
        this->_send_once(batch);
    }
}

void MuDispatcher::put(const TensorBatch &batch) {
    std::lock_guard<std::mutex> lock(this->mtx);
    this->send_queue.push(batch);
    this->cv.notify_one();
}

/*
    MuAttnDispatcher
*/

MuAttnDispatcher::MuAttnDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    std::vector<Channel_t> channels): 
        MuDispatcher(layer_ids, device_id, channels) {
    
}

void MuAttnDispatcher::_send_once(TensorBatch batch) {
    LOG(DEBUG) << "sending a batch." << LEND;

    int n = batch.metadata->shape[0];
    for (int i = 0; i < n;) {
        int j = i + 1;
        int eid = batch.metadata->infos[i].exp_id;
        while (j < n && batch.metadata->infos[j].exp_id == eid)
            j ++;
        auto buf = tensor_at(batch.data, *batch.metadata, i);
        this->_send_batch(
            eid,
            buf,
            batch.metadata->slice(i, j)
        );
        i = j;
    }

    LOG(DEBUG) << "sent a batch." << LEND;
}

/*
    MuExpertDispatcher
*/

MuExpertDispatcher::MuExpertDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    std::vector<Channel_t> channels,
    std::vector<ChannelInfo> channel_infos): 
        MuDispatcher(layer_ids, device_id, channels),
        channel_infos(channel_infos) {
    int max_layer = 0;
    for (auto i: layer_ids)
        max_layer = std::max(i, max_layer);
    this->attn_channel = std::vector<int>(max_layer + 1, -1);

    for (size_t i = 0; i < channels.size(); i ++) {
        // TODO(hogura|20240930): currently, only support #attn_replica=1
        assert(channel_infos[i].attn_layer_ids.size() <= 1);
        if (channel_infos[i].attn_layer_ids.empty()) // a sampler channel
            continue;
        this->attn_channel[
            channel_infos[i].attn_layer_ids[0]
        ] = i;
    }

    LOG(INFO) << "inited MuExpertDispatcher " << device_id << LEND;
}

int MuExpertDispatcher::_get_attn_channel(int req_id, int layer_id) {
    assert(this->attn_channel[layer_id] != -1);
    return this->attn_channel[layer_id];
}

void MuExpertDispatcher::debug_put(TensorBatch batch) {
    _send_once(batch);
}

void MuExpertDispatcher::_send_once(TensorBatch batch) {
    LOG(DEBUG) << "expert " << device_id << " sending a batch" << LEND;
    auto meta = batch.metadata;
    auto layer_id = meta->layer_id;
    std::vector<int> chans;
    for (auto &info: meta->infos) {
        chans.push_back(_get_attn_channel(info.req_id, meta->layer_id));
    }

    auto batches = group_by<int, std::less<int>>(batch.data, *meta, chans, 
        /*on_gpu=*/ !is_embedding_node(device_id));
    LOG(DEBUG) << "grouped channels" << LEND;

    for (auto &[channel, sub_batch]: batches) {
        this->_send_batch(
            channel,
            sub_batch.data,
            *sub_batch.metadata
        );
    }
    LOG(DEBUG) << "expert " << device_id << " sent a batch" << LEND;
}

/*
    MuPool
*/

MuPool::MuPool(
    std::vector<int> layer_ids, 
    int device_id,
    std::vector<Channel_t> channels,
    bool is_attn): 
    MuHelper(layer_ids, device_id, channels), 
    is_attn(is_attn),
    ctx(channels.size()),
    mq(ctx, zmq::socket_type::pull) {
    
    int num_layers = layer_ids.size();
    int max_layer_id = 0;
    for (auto id: layer_ids)
        max_layer_id = std::max(max_layer_id, id);
    this->inner_layer_id = std::vector<int>(max_layer_id + 1);
    this->data_queue = std::vector<std::vector<TensorBatch>>(num_layers);
    for (size_t i = 0; i < num_layers; i ++)
        this->inner_layer_id[layer_ids[i]] = i;

    this->cur_request_count = 0;

    int max_peer_id = 0;
    for (auto c: channels)
        max_peer_id = std::max(max_peer_id, c->get_peer_id());
    this->peer_channels = std::vector<Channel_t>(max_peer_id + 1);
    for (size_t i = 0; i < channels.size(); i ++) {
        int id = channels[i]->get_peer_id();
        assert(this->peer_channels[id].get() == nullptr);
        this->peer_channels[ channels[i]->get_peer_id() ] = channels[i];
    }

    this->tokens_per_layer_ = std::vector<int>(num_layers, 0);
}

void MuPool::run() {
    if (this->channels.empty()) {
        LOG(WARNING) << this->device_id << " has no channels, exit MuPool." << LEND;
        return;
    }
    this->mq.bind(get_zmq_addr(this->device_id));

    auto last = clock();
    auto start = last;

    while (!this->end_flag) {
        LOG(DEBUG) << "fetching a msg ..." << LEND;
        
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_result_t result =
            zmq::recv_multipart(this->mq, std::back_inserter(recv_msgs));
            
        LOG(DEBUG) << "got a msg!" << LEND;
        assert(*result == 2);

        int peer_id = std::stoi(recv_msgs[0].to_string());
        auto metadata = decerealize((char*) recv_msgs[1].data(), recv_msgs[1].size());
        uintptr_t tensor_buf = 0;
        if (metadata->layer_id == 0 && this->is_attn) {
            // Use ZMQ Channel
            tensor_buf = (uintptr_t) std::malloc(
                metadata->num_element() * metadata->get_datatype_size()
            );
        } else {
            // Use NCCL Channel
            tensor_buf = alloc_cuda_tensor(metadata->num_element(), this->device_id);
        }

        LOG(DEBUG) << "calling channel recv from " << peer_id << ", " << *metadata << LEND;
        this->peer_channels[peer_id]->recv(tensor_buf, *metadata);

        if (metadata->layer_id == 0 && this->is_attn) {
            // !NOTE(hogura|20241007): incurs a CPU -> GPU sync here
            tensor_buf = alloc_copy_tensor(
                tensor_buf, 
                metadata->num_element() * metadata->get_datatype_size()
            );
        }

        // TODO: separate sequences into waiting queue and running queue
        int lid = this->inner_layer_id[metadata->layer_id];

        // sync prefill sequences
        /*

        completed_sequences = fill(batch)
        flash_attn_metadata = concat(completed_sequences, decode_tokens from batch)


        1. split batch to prefill and decode tokens
        2. use prefill tokens to compose prefill sequence (sync for prefill sequences) (memcpy, custom kernel)
        3. if a prefill sequence is complete, check if this sequence exists in block table
        4. if not, add them to waiting queue; else add to corresponding running queue with decoding tokens (memcpy)

        waiting_queue, each element is a sequence

        running_queue[layers]

        */
        
        {
            std::lock_guard<std::mutex> lock(this->batch_mutex);
            this->data_queue[lid].push_back((TensorBatch) {
                tensor_buf,
                metadata
            });
        }

        {
            std::lock_guard<std::mutex> lock(this->request_mutex);
            // TODO(hogura|20240930): should modify this `1` with `metadata.num_tokens`
            this->cur_request_count += metadata->num_tokens();
            this->request_cv.notify_all();
        }
    }
}

void MuPool::wait_for_new_requests() {
    LOG(INFO) << "MuPool waiting for new requests" << LEND;
    std::unique_lock<std::mutex> lock(this->request_mutex);
    if (this->cur_request_count > 0) {
        lock.unlock();
        return;
    }
    this->request_cv.wait(lock, [&] { return this->cur_request_count > 0; });
    lock.unlock();
    LOG(INFO) << "MuPool got new requests." << LEND;
}

// the batch_mutex must be used outside this function
int MuPool::tokens_in_layer(int lid) {
    auto &q = this->data_queue[lid];
    int total_tokens = 0;
    for (auto &d: q) {
        total_tokens += d.metadata->num_tokens();
    }
    return total_tokens;
}

std::vector<TensorBatch> MuPool::fetch_largest_batch() {
    // TODO(hogura|20240930): only considering decode first
    LOG(INFO) << "fetching largest batch" << LEND;

    int id = -1;
    size_t max_batch_size = 0;

    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);

        for (size_t i = 0; i < this->data_queue.size(); i ++) {
            int tokens_cur_layer = this->tokens_in_layer(i);
            if (max_batch_size < tokens_cur_layer) {
                max_batch_size = tokens_cur_layer;
                id = i;
            }
        }
    }
    
    if (id == -1) {
        LOG(INFO) << "No available batch" << LEND;
        return {};
    }
    LOG(INFO) << "Fetched " << id << " layer" << LEND;

    std::vector<TensorBatch> result;
    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);

        result = std::move(this->data_queue[id]);
        this->data_queue[id].clear();
    }

    {
        std::lock_guard<std::mutex> lock(this->request_mutex);
        this->cur_request_count -= max_batch_size;
        assert(this->cur_request_count >= 0);
    }

    return result;
}


MuAttentionPool::MuAttentionPool(
    std::vector<int> layer_ids, 
    int device_id,
    std::vector<Channel_t> channels
): MuPool(layer_ids, device_id, channels) {
    int num_layers = layer_ids.size();
    this->is_attn = false;
    this->attn_data_queue = std::vector<std::vector<AttentionBatch>>(num_layers);
}

AttentionBatch MuAttentionPool::pack_attn_batch(uintptr_t data_ptr, metadata_t meta) {
    // for a simple case we consider prefill sequences can only have 1 token,
    // so all sequences in tensor are complete and can be scheduled immediately

    // TODO: deal with incomplete prefill sequences

    auto shape = meta->shape;
    auto dtype = meta->dtype;
    int layer_id = meta->layer_id;

    auto& token_infos = meta->infos;

    int num_tokens = token_infos.size();

    int num_prefill_seqs = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;

    std::vector<int> seq_ids{};
    std::vector<int> prefill_seq_len{};
    std::vector<int> prefill_query_len{};

    for (auto &info: token_infos) {
        if (info.prefill_pos != -1) {
            num_prefill_tokens ++;
            num_prefill_seqs ++;
        } else {
            num_decode_tokens ++;
        }
        int seq_id = info.req_id;
        seq_ids.emplace_back(seq_id);
        prefill_seq_len.emplace_back(1);
        prefill_query_len.emplace_back(1);
    }

    auto attn_meta = std::make_shared<AttentionBatchMetadata> (AttentionBatchMetadata {
        layer_id, shape, dtype,
        num_prefill_seqs,
        num_prefill_tokens,
        num_decode_tokens,
        seq_ids,
        prefill_seq_len,
        prefill_query_len
    });

    return AttentionBatch {data_ptr, attn_meta};
}

void MuAttentionPool::run() {
    if (this->channels.empty()) {
        LOG(WARNING) << this->device_id << " has no channels, exit MuPool." << LEND;
        return;
    }
    this->mq.bind(get_zmq_addr(this->device_id));

    auto last = clock();
    auto start = last;

    while (!this->end_flag) {
        LOG(DEBUG) << "fetching a msg ..." << LEND;
        
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_result_t result =
            zmq::recv_multipart(this->mq, std::back_inserter(recv_msgs));
            
        LOG(DEBUG) << "got a msg!" << LEND;
        assert(*result == 2);

        int peer_id = std::stoi(recv_msgs[0].to_string());
        auto metadata = decerealize((char*) recv_msgs[1].data(), recv_msgs[1].size());
        uintptr_t tensor_buf = 0;
        if (metadata->layer_id == 0 && this->is_attn) {
            // Use ZMQ Channel
            tensor_buf = (uintptr_t) std::malloc(
                metadata->num_element() * metadata->get_datatype_size()
            );
        } else {
            // Use NCCL Channel
            tensor_buf = alloc_cuda_tensor(metadata->num_element(), this->device_id);
        }

        LOG(DEBUG) << "calling channel recv from " << peer_id << ", " << *metadata << LEND;
        this->peer_channels[peer_id]->recv(tensor_buf, *metadata);

        if (metadata->layer_id == 0 && this->is_attn) {
            // !NOTE(hogura|20241007): incurs a CPU -> GPU sync here
            tensor_buf = alloc_copy_tensor(
                tensor_buf, 
                metadata->num_element() * metadata->get_datatype_size()
            );
        }

        // TODO: separate sequences into waiting queue and running queue
        int lid = this->inner_layer_id[metadata->layer_id];

        // sync prefill sequences
        /*

        completed_sequences = fill(batch)
        flash_attn_metadata = concat(completed_sequences, decode_tokens from batch)


        1. split batch to prefill and decode tokens
        2. use prefill tokens to compose prefill sequence (sync for prefill sequences) (memcpy, custom kernel)
        3. if a prefill sequence is complete, check if this sequence exists in block table
        4. if not, add them to waiting queue; else add to corresponding running queue with decoding tokens (memcpy)

        waiting_queue, each element is a sequence

        running_queue[layers]

        */            
        
        auto attn_batch = pack_attn_batch(tensor_buf, metadata);
        int batched_tokens = attn_batch.metadata->num_decode_tokens + attn_batch.metadata->num_prefill_tokens;
        
        {
            std::lock_guard<std::mutex> lock(this->batch_mutex);

            int &tokens_cur_layer = this->tokens_per_layer_[lid];
            tokens_cur_layer += batched_tokens;
            if (tokens_cur_layer > this->largest_batch_size_) {
                this->largest_batch_size_ = tokens_cur_layer;
                this->largest_batch_layer_id_ = lid;
            }

            this->attn_data_queue[lid].push_back(attn_batch);
        }

        {
            std::lock_guard<std::mutex> lock(this->request_mutex);
            // TODO(hogura|20240930): should modify this `1` with `metadata.num_tokens`
            this->cur_request_count += batched_tokens;
            this->request_cv.notify_all();
        }
    }
}

// the batch_mutex must be used outside this function
int MuAttentionPool::tokens_in_layer(int lid) {
    auto &q = this->attn_data_queue[lid];
    int num_tokens = 0;
    for (auto &d: q) {
        num_tokens += d.metadata->num_prefill_tokens + d.metadata->num_decode_tokens;
    }
    return num_tokens;
}

std::vector<AttentionBatch> MuAttentionPool::fetch_largest_batch() {
    // TODO(hogura|20240930): only considering decode first

    LOG(INFO) << "fetching largest batch" << LEND;
    int layer_id = -1;
    int batched_tokens = 0;
    std::vector<AttentionBatch> result{};
    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);

        layer_id = this->largest_batch_layer_id_;
        batched_tokens = this->largest_batch_size_;

        if (layer_id == -1) {
            LOG(INFO) << "No available batch" << LEND;
            return {};
        }

        LOG(INFO) << "Fetched " << layer_id << " layer" << LEND;

        result = std::move(this->attn_data_queue[layer_id]);
        this->attn_data_queue[layer_id].clear();

        int num_layers = this->layer_ids.size();

        this->largest_batch_size_ = 0;
        this->largest_batch_layer_id_ = -1;
        for (int i = 0; i < num_layers; i++) {
            int num_tokens = tokens_in_layer(i);
            if (num_tokens > this->largest_batch_size_) {
                this->largest_batch_size_ = num_tokens;
                this->largest_batch_layer_id_ = i;
            }
        }

    }

    {
        std::lock_guard<std::mutex> lock(this->request_mutex);
        this->cur_request_count -= batched_tokens;
        assert(this->cur_request_count >= 0);
    }

    return result;
}