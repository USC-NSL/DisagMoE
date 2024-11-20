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
#include "cuda_utils.h"

#include "zmq.hpp"
#include "zmq_addon.hpp"

#include <cereal/archives/binary.hpp>

// MuHelper

MuHelper::MuHelper(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    layer_ids(layer_ids), device_id(device_id), channels(channels), end_flag(false) {
        DMOE_LOG(INFO) << "init muhelper@" << device_id << LEND;
    }

void MuHelper::start() {
    DMOE_LOG(INFO) << "muhelper@" << device_id << " start" << LEND;
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

MuDispatcher::MuDispatcher(std::vector<int> layer_ids, int device_id, 
                           ParallelConfig cfg, std::vector<Channel_t> channels,
                           const std::vector<bool> &is_group_channels): 
    MuHelper(layer_ids, device_id, channels), 
    peer_ctx(channels.size()),
    peer_mq(channels.size()),
    cfg(cfg),
    is_group_channels(is_group_channels) {
    sprintf(this->device_id_str, "%d", this->device_id);
    if (is_group_channels.empty())
        this->is_group_channels.resize(channels.size(), false);
    ASSERT(this->is_group_channels.size() == channels.size());
    group_channels.resize(channels.size());
    for (int i = 0; i < channels.size(); i ++) {
        peer_ctx[i] = zmq::context_t(1);
        peer_mq[i] = zmq::socket_t(peer_ctx[i], zmq::socket_type::push);
        if (_is_group_channel(i)) {
            group_channels[i] = std::dynamic_pointer_cast<NcclGroupChannel>(channels[i]);
        }
    }
}

bool MuDispatcher::_is_group_channel(int cid) const {
    return is_group_channels[cid];
}

void MuDispatcher::_send_batch(int cid, uintptr_t buf, const Metadata& meta) {
    tx_range _{"MuDispatcher::_send_batch"};
    // DMOE_LOG(DEBUG) << "sending batch to channel " << cid << " current device: " << this->device_id_str << LEND;

    if (!_is_group_channel(cid)) {
        auto data = cerealize(std::make_shared<Metadata>(meta));
        this->peer_mq[cid].send(zmq::str_buffer(this->device_id_str), zmq::send_flags::sndmore);
        this->peer_mq[cid].send(zmq::buffer(data.c_str(), data.size()));
        this->channels[cid]->send(buf, meta);
    } else {
        this->group_channels[cid]->send_metadata(meta);
        this->group_channels[cid]->send(buf, meta);
    }

    // DMOE_LOG(DEBUG) << "sent batch to channel " << cid << LEND;
}

void MuDispatcher::run() {
    for (int i = 0; i < this->channels.size(); i ++)
        this->peer_mq[i].connect(get_zmq_addr(this->channels[i]->get_peer_id()));

    DMOE_LOG(DEBUG) << "running mudispatcher@" << this->device_id << LEND;
    while (!this->end_flag) {
        // DMOE_LOG(WARNING) << "waiting for new dispatching request ..." << LEND;
        std::unique_lock<std::mutex> lock(this->mtx);
        this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
        // DMOE_LOG(WARNING) << "Got a request !!!" << LEND;
        auto pr = this->send_queue.front();
        auto batch = pr.first;
        int rank = pr.second;
        this->send_queue.pop();
        this->_send_once(batch);
    }
}

void MuDispatcher::put(TensorBatch batch, int rank) {
    std::lock_guard<std::mutex> lock(this->mtx);
    batch.data = batch.data.clone().detach();
    this->send_queue.push(std::make_pair(batch, rank));
    this->cv.notify_one();
}

/*
    MuAttnDispatcher
*/

MuAttnDispatcher::MuAttnDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    ParallelConfig cfg,
    std::vector<Channel_t> channels,
    const std::vector<ChannelInfo> &out_channel_infos): 
        MuDispatcher(layer_ids, device_id, cfg, channels) {
    int max_layer_id = 0;
    max_exp_id = 0;
    for (auto &info: out_channel_infos) {
        for (auto pr: info.expert_ids) {
            max_exp_id = std::max(max_exp_id, pr.second);
            max_layer_id = std::max(max_layer_id, pr.first);
        }
    }
    max_exp_id ++;
    DMOE_LOG(INFO) << "max_layer_id " << max_layer_id << ", max_exp_id " << max_exp_id << LEND;
    exp_channels.resize(_encode(max_layer_id + 1, 0), -1);

    for (int i = 0; i < channels.size(); i ++) {
        for (auto exp_id: out_channel_infos[i].expert_ids) {
            int id = _encode(exp_id.first, exp_id.second);
            DMOE_LOG(WARNING) << "exp_id " << exp_id.first << " " << exp_id.second << " " << id << LEND;
            exp_channels[id] = i;
        }
    }
}

inline int MuAttnDispatcher::_get_rank(int exp_id) const {
    return exp_id / cfg.n_exp_per_rank;
}

inline int MuAttnDispatcher::_encode(int exp_layer_id, int exp_id) const {
    return exp_layer_id * this->max_exp_id + _get_rank(exp_id);
}

void MuAttnDispatcher::_send_once(TensorBatch batch) {
    tx_range _{"MuAttnDispatcher::_send_once"};
    // DMOE_LOG(DEBUG) << "sending a batch." << LEND;
    // DMOE_LOG(DEBUG) << "shape size: " << batch.metadata->shape.size()
    //            << " info size: " << batch.metadata->infos.size() << LEND;

    int n = batch.metadata->shape[0];
    int lid = batch.metadata->layer_id;
    for (int i = 0; i < n;) {
        int j = i + 1;
        int ep_rank = _get_rank(batch.metadata->exp_ids[i]);
        while (j < n && _get_rank(batch.metadata->exp_ids[j]) == ep_rank)
            j ++;
        ASSERT(ep_rank >= 0);
        int cid = _encode(lid, batch.metadata->exp_ids[i]);
        if (i == 0 && j == n) {
            // a faster path
            this->_send_batch(
                this->exp_channels[cid],
                (uintptr_t)batch.data.data_ptr(),
                *batch.metadata
            );
            break;
        }

        auto buf = tensor_at((uintptr_t)batch.data.data_ptr(), *batch.metadata, i);
        this->_send_batch(
            this->exp_channels[cid],
            buf,
            batch.metadata->slice(i, j)
        );
        i = j;
    }

    // DMOE_LOG(DEBUG) << "sent a batch." << LEND;
}

/*
    MuExpertDispatcher
*/

MuExpertDispatcher::MuExpertDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    ParallelConfig cfg,
    std::vector<Channel_t> channels,
    std::vector<ChannelInfo> channel_infos,
    const std::vector<bool> &is_group_channels): 
        MuDispatcher(layer_ids, device_id, cfg, channels, is_group_channels),
        channel_infos(channel_infos) {
    int max_layer = -1;
    for (auto info: channel_infos)
        for (int i: info.attn_layer_ids)
            max_layer = std::max(i, max_layer);
    this->attn_channel = std::vector<int>(max_layer + 1, 0);

    for (size_t i = 0; i < channels.size(); i ++) {
        // TODO(hogura|20240930): currently, only support #attn_replica=1
        if (channel_infos[i].attn_layer_ids.empty()) {// a sampler channel 
            this->sampler_channel_id = i;
            continue;
        }
        for (int j: channel_infos[i].attn_layer_ids) {
            ASSERT(!this->attn_channel[j]);
            this->attn_channel[j] = i;
        }
    }

    DMOE_LOG(INFO) << "inited MuExpertDispatcher " << device_id << LEND;
}

int MuExpertDispatcher::_get_attn_channel(int req_id, int layer_id) {
    // DMOE_LOG(DEBUG) << "layer_id: " << layer_id << " attn_chan.size: " << attn_channel.size() << LEND;
    return layer_id < this->attn_channel.size() ? this->attn_channel[layer_id] : sampler_channel_id;
}

void MuExpertDispatcher::debug_put(TensorBatch batch) {
    _send_once(batch);
}

void MuExpertDispatcher::_send_once(TensorBatch batch) {
    tx_range _{"MuExpertDispatcher::_send_once"};

    // DMOE_LOG(DEBUG) << "expert " << device_id << " sending a batch" << LEND;
    auto meta = batch.metadata;
    auto layer_id = meta->layer_id;
    std::vector<int> chans;
    for (int req_id: meta->req_ids) {
        chans.push_back(_get_attn_channel(req_id, layer_id));
    }

    auto batches = group_by<int, std::less<int>>(batch.data, *meta, chans, 
        /*on_gpu=*/ !is_embedding_node(device_id));
    // DMOE_LOG(DEBUG) << "grouped channels" << LEND;

    for (auto &sub_batch: batches) {
        auto &channel = std::get<0>(sub_batch);
        auto &tensor = std::get<1>(sub_batch);
        this->_send_batch(
            channel,
            (uintptr_t)tensor.data_ptr(),
            std::get<2>(sub_batch)
        );
    }
    // DMOE_LOG(DEBUG) << "expert " << device_id << " sent a batch" << LEND;
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
        ASSERT(this->peer_channels[id].get() == nullptr);
        this->peer_channels[ channels[i]->get_peer_id() ] = channels[i];
    }

    this->tokens_per_layer_ = std::vector<int>(num_layers, 0);
}

void MuPool::recv_metadata(int &peer_id, metadata_t &meta) {
    // DMOE_LOG(DEBUG) << "fetching a msg ..." << LEND;
        
    std::vector<zmq::message_t> recv_msgs;
    zmq::recv_result_t result =
        zmq::recv_multipart(this->mq, std::back_inserter(recv_msgs));
        
    // DMOE_LOG(DEBUG) << "got a msg!" << LEND;
    ASSERT(*result == 2);

    peer_id = std::stoi(recv_msgs[0].to_string());
    meta = decerealize<Metadata>((char*) recv_msgs[1].data(), recv_msgs[1].size());
}

void MuPool::recv_tensor(int peer_id, uintptr_t tensor_buf, metadata_t &meta) {
    // DMOE_LOG(DEBUG) << "peer_id " << peer_id << " channelsize " << this->peer_channels.size() << LEND;
    ASSERT(0 <= peer_id && peer_id < this->peer_channels.size());
    ASSERT(this->peer_channels[peer_id].get() != nullptr);
    ASSERT(meta.get() != nullptr);
    ASSERT(tensor_buf != 0);
    this->peer_channels[peer_id]->recv(tensor_buf, *meta);
}

void MuPool::process_batch(torch::Tensor tensor, metadata_t &meta, bool send_from_zmq) {
    // TODO(hogura|20241014): sync prefill sequences
    /*
    TODO(shaoyuw|20241011): separate sequences into waiting queue and running queue
    completed_sequences = fill(batch)
    flash_attn_metadata = concat(completed_sequences, decode_tokens from batch)


    1. split batch to prefill and decode tokens
    2. use prefill tokens to compose prefill sequence (sync for prefill sequences) (memcpy, custom kernel)
    3. if a prefill sequence is complete, check if this sequence exists in block table
    4. if not, add them to waiting queue; else add to corresponding running queue with decoding tokens (memcpy)

    waiting_queue, each element is a sequence

    running_queue[layers]

    */
    int lid = this->inner_layer_id[meta->layer_id];
    int num_tokens = meta->num_tokens();

    // {
    //     std::lock_guard<std::mutex> lock(this->request_mutex);
    //     this->cur_request_count += num_tokens;
    //     this->request_cv.notify_all();
    // }
    
    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);
        this->data_queue[lid].push_back((TensorBatch) {tensor, meta});
        int &tokens_cur_layer = this->tokens_per_layer_[lid];
        tokens_cur_layer += num_tokens;
        if (tokens_cur_layer > this->largest_batch_size_) {
            this->largest_batch_size_ = tokens_cur_layer;
            this->largest_batch_layer_id_ = lid;
        }

    }
}

void MuPool::run() {
    if (this->channels.empty()) {
        DMOE_LOG(WARNING) << this->device_id << " has no channels, exit MuPool." << LEND;
        return;
    }
    this->mq.bind(get_zmq_addr(this->device_id));

    auto last = clock();
    auto start = last;

    DMOE_LOG(DEBUG) << "Running pool@" << this->device_id << LEND;
    while (!this->end_flag) {
        int peer_id;
        metadata_t meta;

        MuPool::recv_metadata(peer_id, meta);

        torch::Tensor tensor = torch::empty(
            {meta->num_tokens(), meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );

        MuPool::recv_tensor(peer_id, (uintptr_t)tensor.data_ptr(), meta);

        this->process_batch(tensor, meta,  /*send_from_zmq=*/ true);
    }
}

void MuPool::wait_for_new_requests() {
    DMOE_LOG(INFO) << "MuPool waiting for new requests" << LEND;
    std::unique_lock<std::mutex> lock(this->request_mutex);
    if (this->cur_request_count > 0) {
        lock.unlock();
        return;
    }
    this->request_cv.wait(lock, [&] { return this->cur_request_count > 0; });
    lock.unlock();
    DMOE_LOG(INFO) << "MuPool got new requests." << LEND;
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

void MuPool::maintain_largest_batch() {
    // !NOTE(hogura|20241106): when calling this function, a lock is required!

    this->largest_batch_size_ = 0;
    this->largest_batch_layer_id_ = -1;
    for (int i = 0; i < tokens_per_layer_.size(); i++) {
        int num_tokens = this->tokens_per_layer_[i];
        if (num_tokens > this->largest_batch_size_) {
            this->largest_batch_size_ = num_tokens;
            this->largest_batch_layer_id_ = i;
        }
    }
}

std::vector<TensorBatch> MuPool::fetch_largest_batch() {
    // TODO(hogura|20240930): only considering decode first
    // DMOE_LOG(INFO) << "fetching largest batch" << LEND;

    int id = -1;
    int max_batch_size = 0;

    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);

        id = this->largest_batch_layer_id_;

        if (id == -1) {
            // DMOE_LOG(INFO) << "No available batch" << LEND;
            return {};
        }
    }
    
    if (id == -1) {
        // DMOE_LOG(INFO) << "No available batch" << LEND;
        return {};
    }
    // DMOE_LOG(INFO) << "Fetched " << id << "-th layer" << LEND;

    std::vector<TensorBatch> result;
    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);

        result = std::move(this->data_queue[id]);
        this->data_queue[id].clear();
        max_batch_size = this->largest_batch_size_;

        tokens_per_layer_[id] = 0;
        maintain_largest_batch();
    }

    // {
    //     std::lock_guard<std::mutex> lock(this->request_mutex);
    //     this->cur_request_count -= max_batch_size;
    //     ASSERT(this->cur_request_count >= 0);
    // }

    return result;
}


MuAttentionPool::MuAttentionPool(
    std::vector<int> layer_ids, 
    int device_id,
    std::vector<Channel_t> channels,
    std::vector<int> device_group_ids,
    Channel_t group_comm
): MuPool(layer_ids, device_id, channels, true), 
    device_group_ids(device_group_ids),
    group_comm(group_comm.get() != nullptr ? std::dynamic_pointer_cast<NcclGroupChannel>(group_comm) : nullptr) {
    int num_layers = layer_ids.size();
    this->attn_data_queue = std::vector<std::vector<AttentionBatch>>(num_layers);
}

void MuAttentionPool::run() {
    pool_thread = std::thread([&]() {
        MuPool::run();
    });

    if (device_group_ids.size() <= 1) {
        DMOE_LOG(WARNING) << "No group channel is needed in MuAttnPool::run." << LEND;
        return;
    }

    if (device_group_ids.size() > 1 && device_group_ids[0] != device_id) {
        DMOE_LOG(INFO) << "Running ATTN Worker pool (intra-group)" << LEND;
        group_threads.emplace_back(std::thread([&]() {
            at::cuda::CUDAStream c10_stream = at::cuda::getCurrentCUDAStream(0);
            at::cuda::CUDAStreamGuard guard(c10_stream);
            while (!end_flag) {
                // DMOE_LOG(DEBUG) << "Worker AttnPool fetching metadata ..." << LEND;
                Metadata meta;
                group_comm->recv_metadata(meta);

                torch::Tensor tensor = torch::empty(
                    {meta.num_tokens(), meta.token_hidden_dim()}, 
                    torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
                );

                DMOE_LOG(DEBUG) << "Worker AttnPool fetched result:" << meta << LEND;
                group_comm->recv((uintptr_t) tensor.data_ptr(), meta);
                DMOE_LOG(DEBUG) << "Worker AttnPool broadcast finished" << LEND;
                auto t_meta = std::make_shared<Metadata>(meta);
                process_batch(tensor, t_meta, /*send_from_zmq=*/ false);
            }
        }));
    }

    DMOE_LOG(INFO) << "Running ATTN Driver/Worker pool (inter-group)" << LEND;
    for (auto &c: this->channels) {
        if (is_embedding_node(c->get_peer_id()))
            continue;
        // if not embedding, then must be an expert dispatcher, which is a group channel
        ASSERT(c.get() != nullptr);
        auto group_c = std::dynamic_pointer_cast<NcclGroupChannel>(c);
        ASSERT(group_c.get() != nullptr);
        group_threads.emplace_back(std::thread(
            [&](std::shared_ptr<NcclGroupChannel> c) {
                // recv messages from multiple dispatchers
                at::cuda::CUDAStream c10_stream = at::cuda::getCurrentCUDAStream(0);
                at::cuda::CUDAStreamGuard guard(c10_stream);

                while (!end_flag) {
                    // DMOE_LOG(DEBUG) << "AttnPool fetching metadata ..." << LEND;
                    Metadata meta;
                    c->recv_metadata(meta);

                    torch::Tensor tensor = torch::empty(
                        {meta.num_tokens(), meta.token_hidden_dim()}, 
                        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
                    );

                    DMOE_LOG(DEBUG) << "AttnPool fetched: " << meta << LEND;
                    c->recv((uintptr_t)tensor.data_ptr(), meta);
                    DMOE_LOG(DEBUG) << "AttnPool broadcast finished" << LEND;

                    auto meta_t = std::make_shared<Metadata>(meta);

                    // When using large_comm, all tensors are sent to the workers
                    // Only using ZMQ & layer_id == 0, the tensors are required to be broadcast through small_comm
                    // See MuAttentionPool::process_batch
                    process_batch(tensor, meta_t, /*send_from_zmq=*/ false);
                }
            }, group_c
        ));
    }
}

void MuAttentionPool::terminate() {
    MuPool::terminate();
    pool_thread.join();
}

AttentionBatch MuAttentionPool::pack_attn_batch(torch::Tensor tensor, metadata_t meta) {
    // for a simple case we consider prefill sequences can only have 1 token,
    // so all sequences in tensor are complete and can be scheduled immediately

    // TODO: deal with incomplete prefill sequences

    auto shape = meta->shape;
    auto dtype = meta->dtype;
    int layer_id = meta->layer_id;

    int num_tokens = meta->req_ids.size();

    int num_prefill_seqs = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;

    std::vector<int> seq_ids{};
    std::vector<int> prefill_seq_len{};
    std::vector<int> prefill_query_len{};

    for (int i = 0; i < meta->req_ids.size(); i ++) {
        if (meta->prefill_poss[i] != -1) {
            num_prefill_tokens ++;
            num_prefill_seqs ++;
        } else {
            num_decode_tokens ++;
        }
        seq_ids.emplace_back(meta->req_ids[i]);
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

    return AttentionBatch {tensor, attn_meta};
}

void MuAttentionPool::process_batch(torch::Tensor tensor, metadata_t &meta, bool send_from_zmq) {
    // DMOE_LOG(DEBUG) << "AttnPool processing batch: " << meta << LEND;
    if (send_from_zmq && meta->layer_id == 0 && group_comm.get() != nullptr) {
        // since only driver can have the pool, we can send the data from layer 0 to other workers here.
        // NOTE(hogura|20241110): group_comm is only used when send_from_zmq, so it should be thread-safe
        DMOE_LOG(DEBUG) << "Broadcasting attn batch to workers" << LEND;
        group_comm->send_metadata(*meta);
        group_comm->send((uintptr_t) tensor.data_ptr(), *meta);
        DMOE_LOG(DEBUG) << "Broadcast finished." << LEND;
    }

    int lid = this->inner_layer_id[meta->layer_id];
    auto attn_batch = pack_attn_batch(tensor, meta);
    int batched_tokens = attn_batch.metadata->num_decode_tokens + attn_batch.metadata->num_prefill_tokens;

    // {
    //     std::lock_guard<std::mutex> lock(this->request_mutex);
    //     this->cur_request_count += batched_tokens;
    //     this->request_cv.notify_all();
    //     DMOE_LOG(WARNING) << "add cur_request_count:" << cur_request_count << " " << batched_tokens << LEND;
    // }
    
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
}

// the batch_mutex must be used outside this function
int MuAttentionPool::tokens_in_layer(int lid) {
    auto &q = this->attn_data_queue[lid];
    int num_tokens = 0;
    for (auto &d: q) {
        num_tokens += d.metadata->num_prefill_tokens + d.metadata->num_decode_tokens;
        DMOE_LOG(DEBUG) << "tokens_in_layer #" << lid << " " 
            << d.metadata->num_prefill_tokens << " " 
            << d.metadata->num_decode_tokens << LEND;
    }
    return num_tokens;
}

std::vector<AttentionBatch> MuAttentionPool::fetch_largest_batch(int *selected_layer_id) {
    // TODO(hogura|20240930): only considering decode first

    // DMOE_LOG(INFO) << "fetching largest batch" << LEND;
    int layer_id = -1;
    int batched_tokens = 0;
    std::vector<AttentionBatch> result{};
    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);

        layer_id = this->largest_batch_layer_id_;

        if (layer_id == -1) {
            // DMOE_LOG(INFO) << "No available batch" << LEND;
            if (selected_layer_id)
                *selected_layer_id = -1;
            return {};
        }
        
        batched_tokens = this->largest_batch_size_;

        result = std::move(this->attn_data_queue[layer_id]);
        this->attn_data_queue[layer_id].clear();
        this->tokens_per_layer_[layer_id] = 0;

        int num_layers = this->layer_ids.size();

        maintain_largest_batch();
    }

    // DMOE_LOG(DEBUG) << "Fetched " << layer_id << " layer with #tokens=" << batched_tokens << LEND;

    // {
    //     std::lock_guard<std::mutex> lock(this->request_mutex);
    //     DMOE_LOG(WARNING) << "del cur_request_count:" << cur_request_count << " " << batched_tokens << LEND;
    //     this->cur_request_count -= batched_tokens;
    //     DMOE_LOG(DEBUG) << "cur request count: " << cur_request_count << LEND;
    //     ASSERT(this->cur_request_count >= 0);
    // }

    if (selected_layer_id)
        *selected_layer_id = layer_id;
    return result;
}

std::vector<AttentionBatch> MuAttentionPool::fetch_batch_from(
    int layer_id, std::set<int> &seq_ids) {

    // DMOE_LOG(WARNING) << "fetching " << seq_ids.size() << " batches from worker's layer " << layer_id << LEND;

    // wait until the data_queue has enough batches
    for (bool flag = false; !flag;) {
        std::lock_guard<std::mutex> lock(this->batch_mutex);
        int sum = 0;
        for (auto &batch: this->attn_data_queue[layer_id]) {
            int id = batch.metadata->seq_ids[0];
            if (seq_ids.find(id) != seq_ids.end()) {
                // !NOTE(hogura|20241119): here we make an asumption:
                // the batch in the attn_data_queue must not be merged,
                // each batch send to the driver must be sent to the worker as-is.
                sum += batch.metadata->seq_ids.size();
            }
        }
        ASSERT(sum <= seq_ids.size());
        if (sum == seq_ids.size()) {
            flag = true;
        }
    }

    // DMOE_LOG(WARNING) << "should have fetched " << seq_ids.size() << " seq_ids from worker's layer " << layer_id << LEND;

    std::lock_guard<std::mutex> lock(this->batch_mutex);
    ASSERT(layer_id >= 0 && layer_id < this->attn_data_queue.size());

    std::vector<AttentionBatch> result, remains;
    for (auto &batch: this->attn_data_queue[layer_id]) {
        int id = batch.metadata->seq_ids[0];
        if (seq_ids.find(id) != seq_ids.end()) {
            result.emplace_back(batch);
        } else {
            remains.emplace_back(batch);
        }
    }

    int num_tokens = 0;
    for (auto &batch: result)
        num_tokens += batch.metadata->num_prefill_tokens + batch.metadata->num_decode_tokens;
    this->tokens_per_layer_[layer_id] -= num_tokens;
    this->attn_data_queue[layer_id] = remains;

    maintain_largest_batch();

    // DMOE_LOG(WARNING) << "fetched " << result.size() << " batches and " << num_tokens << " tokens from worker's layer " << layer_id << LEND;

    return result;
}