#include <condition_variable>
#include <cstdlib>
#include <string>
#include <mutex>
#include <queue>
#include <ctime>
#include <utility>

#include "distributed.hpp"
#include "datatypes.hpp"
#include "muhelper.h"
#include "comm.h"
#include "utils.hpp"
#include "logging.h"
#include "constants.h"
#include "cuda_utils.h"
#include "profiler.hpp"

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
            Recorder::create();
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
    for (int i = 0; i < this->channels.size(); i ++) {
        this->peer_mq[i].connect(get_zmq_addr(this->channels[i]->get_peer_id(), true, -1, this->peer_zmq_port_offset));
    }

    // DMOE_LOG(DEBUG) << "running mudispatcher@" << this->device_id << LEND;
    while (!this->end_flag) {
        // DMOE_LOG(WARNING) << "waiting for new dispatching request ..." << LEND;
        TensorBatch batch;
        {
            // Fetch a batch from the queue, lock required (for the send_queue).
            std::unique_lock<std::mutex> lock(this->mtx);
            this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
            // DMOE_LOG(WARNING) << "Got a request !!!" << LEND;
            auto pr = this->send_queue.front();
            batch = pr.first;
            // pr.second(i.e. rank) is not used for now
            this->send_queue.pop();
        }
        // Send the batch, no lock required, since send_queue won't be changed.
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
    this->peer_zmq_port_offset = 0;
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
    exp_channels.resize((max_layer_id + 1) * max_exp_id, -1);

    // get expert ranks
    _inner_expert_ranks.resize(max_layer_id + 1);
    for (int i = 0; i <= max_layer_id; i ++)
        _inner_expert_ranks[i].resize(max_exp_id + 1, -1);
    for (auto &tuple: cfg.expert_ranks) {
        int layer_id = std::get<0>(tuple);
        int exp_id = std::get<1>(tuple);
        int rank = std::get<2>(tuple);
        _inner_expert_ranks[layer_id][exp_id] = rank;
        ASSERT(rank < max_exp_id);
    }

    // get expert channels
    for (int i = 0; i < channels.size(); i ++) {
        for (auto exp_id: out_channel_infos[i].expert_ids) {
            int id = _encode(exp_id.first, exp_id.second);
            exp_channels[id] = i;
        }
    }
}

inline int MuAttnDispatcher::_get_rank(int exp_layer_id, int exp_id) const {
    ASSERT(_inner_expert_ranks[exp_layer_id][exp_id] >= 0);
    return _inner_expert_ranks[exp_layer_id][exp_id];
}

inline int MuAttnDispatcher::_encode(int exp_layer_id, int exp_id) const {
    return exp_layer_id * this->max_exp_id + _get_rank(exp_layer_id, exp_id);
}

void MuAttnDispatcher::_send_once(TensorBatch batch) {
    tx_range _{"MuAttnDispatcher::_send_once"};
    // DMOE_LOG(DEBUG) << "attn " << this->device_id << " sending a batch: " << *batch.metadata << LEND;
    // DMOE_LOG(DEBUG) << "shape size: " << batch.metadata->shape.size()
    //            << " info size: " << batch.metadata->infos.size() << LEND;

    int n = batch.metadata->shape[0];
    int lid = batch.metadata->layer_id;
    for (int i = 0; i < n;) {
        int j = i + 1;
        int ep_rank = _get_rank(lid, batch.metadata->exp_ids[i]);
        while (j < n && _get_rank(lid, batch.metadata->exp_ids[j]) == ep_rank)
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

        auto sliced_meta = batch.metadata->slice(i, j);

        auto buf = tensor_at((uintptr_t)batch.data.data_ptr(), *batch.metadata, i);
        this->_send_batch(
            this->exp_channels[cid],
            buf,
            sliced_meta
        );
        i = j;

        // DMOE_LOG(INFO) << "attn send a batch to expert: " << sliced_meta << LEND;
    }

    // DMOE_LOG(DEBUG) << "attn sent a batch." << LEND;
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
    this->peer_zmq_port_offset = 1;
    int max_layer = -1;
    for (auto info: channel_infos)
        for (int i: info.attn_layer_ids)
            max_layer = std::max(i, max_layer);

    // attn_channel[layer_id][dp_rank]
    this->attn_channel.resize(max_layer + 1, {});
    for (int i = 0; i <= max_layer; i ++)
        this->attn_channel[i].resize(cfg.dp, -1);

    for (size_t i = 0; i < channels.size(); i ++) {
        if (channel_infos[i].attn_layer_ids.empty()) {// a sampler channel 
            this->sampler_channel_id = i;
            continue;
        }
        int dp_rank = channel_infos[i].attn_dp_rank;
        for (int j = 0; j < channel_infos[i].attn_layer_ids.size(); j ++) {
            int lid = channel_infos[i].attn_layer_ids[j];
            DMOE_LOG(DEBUG) << "channel " << i << " attn_layer_id " << lid << " dp_rank " << dp_rank << LEND;
            ASSERT(this->attn_channel[lid][dp_rank] == -1);
            this->attn_channel[lid][dp_rank] = i;
        }
    }

    DMOE_LOG(INFO) << "inited MuExpertDispatcher " << device_id << LEND;
}

int MuExpertDispatcher::_get_attn_channel(int layer_id, int rank) {
    // DMOE_LOG(DEBUG) << "layer_id: " << layer_id << " attn_chan.size: " << attn_channel.size() << LEND;
    return layer_id < this->attn_channel.size() ? this->attn_channel[layer_id][rank] : sampler_channel_id;
}

void MuExpertDispatcher::debug_put(TensorBatch batch) {
    _send_once(batch);
}

void MuExpertDispatcher::_send_once(TensorBatch batch) {
    tx_range _{"MuExpertDispatcher::_send_once"};
    auto meta = batch.metadata;
    auto layer_id = meta->layer_id;

    // DMOE_LOG(DEBUG) << "expert " << device_id << " sending a batch: " << *meta << ", n_ele=" << batch.data.numel()  << LEND;
    ASSERT(batch.data.sizes()[0] == meta->shape[0]);
    ASSERT(batch.data.sizes()[1] == meta->shape[1]);

    // DP_SIZE == 1, or a sampler channel
    if (this->attn_channel[0].size() == 1 || layer_id >= this->attn_channel.size()) {
        this->_send_batch(
            _get_attn_channel(layer_id, 0),
            (uintptr_t) batch.data.data_ptr(),
            *meta
        );
    } else {
        auto &channels = this->attn_channel[layer_id];
        for (int i = 0, j = 1, n = meta->attn_dp_ranks.size(); i < n; i = j) {
            int rank = meta->attn_dp_ranks[i];
            ASSERT(0 <= rank && rank < channels.size());
            while (j < n && meta->attn_dp_ranks[j] == rank)
                j ++;

            // a faster path
            if (i == 0 && j == n) {
                this->_send_batch(
                    channels[rank],
                    (uintptr_t) batch.data.data_ptr(),
                    *meta
                );
            } else {
                auto buf = tensor_at((uintptr_t) batch.data.data_ptr(), *batch.metadata, i);
                this->_send_batch(
                    channels[rank],
                    buf,
                    batch.metadata->slice(i, j)
                );
            }
        }
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
    mq(ctx, zmq::socket_type::pull),
    max_batch_size(MAX_BATCH_SIZE) {
    this->local_zmq_port_offset = 0;
    int num_layers = layer_ids.size();
    int max_layer_id = 0;
    for (auto id: layer_ids)
        max_layer_id = std::max(max_layer_id, id);
    this->layer_id_P2V = std::vector<int>(max_layer_id + 1);
    this->layer_id_V2P = std::vector<int>(num_layers);

    this->data_queue = std::vector<std::vector<TensorBatch>>(num_layers);
    for (size_t i = 0; i < num_layers; i ++) {
        this->layer_id_P2V[layer_ids[i]] = i;
        this->layer_id_V2P[i] = layer_ids[i];
    }

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
    // DMOE_LOG(INFO) << "Expert pool receive metadata: " << *meta << LEND;
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
    int lid = this->layer_id_P2V[meta->layer_id];
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
    this->mq.bind(get_zmq_addr(this->device_id, true, -1, this->local_zmq_port_offset));

    auto last = clock();
    auto start = last;

    // DMOE_LOG(DEBUG) << "Running pool@" << this->device_id << LEND;
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
    // DMOE_LOG(INFO) << "MuPool waiting for new requests" << LEND;
    std::unique_lock<std::mutex> lock(this->request_mutex);
    if (this->cur_request_count > 0) {
        lock.unlock();
        return;
    }
    this->request_cv.wait(lock, [&] { return this->cur_request_count > 0; });
    lock.unlock();
    // DMOE_LOG(INFO) << "MuPool got new requests." << LEND;
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

std::vector<int> MuPool::get_pool_snapshot() {
    std::lock_guard<std::mutex> lock(this->batch_mutex);
    return this->tokens_per_layer_;
}

template<class DataBatch>
int schedule_with_limit_dp(std::vector<DataBatch> &data_list,
                         std::vector<DataBatch> &results,
                         int max_batch_size,
                         bool allow_sliced = false) {
    ASSERT(!allow_sliced);

    std::vector<int> dp(max_batch_size + 1, 0), pre(max_batch_size + 1, -1), marked(data_list.size(), 0);
    
    // formulating this scheduling into a Knapsack problem
    dp[0] = 1;
    for (int i = 0; i < data_list.size(); i ++) {
        for (int j = max_batch_size; j >= data_list[i].metadata->num_tokens(); j --) {
            if (dp[j - data_list[i].metadata->num_tokens()] && !dp[j]) {
                dp[j] = 1;
                pre[j] = i;
            }
        }
    }

    // select the schedule results
    int cur = max_batch_size;
    while (!dp[cur] && cur > 0)
        cur --;
    if (cur == 0) {
        DMOE_LOG(WARNING) << "MaxBatchSize=" << max_batch_size << "; Current DataBatch sizes: ";
        for (auto &d: data_list)
            std::cerr << d.metadata->num_tokens() << " ";
        std::cerr << LEND;
    }
    int num_tokens = cur;
    while (cur > 0) {
        results.push_back(data_list[pre[cur]]);
        marked[pre[cur]] = 1;
        cur -= data_list[pre[cur]].metadata->num_tokens();
    };

    std::vector<DataBatch> data_queue;
    int max_num_tokens = 0;
    int max_batch_id = -1;
    // restore the data_queue
    for (int i = 0; i < data_list.size(); i ++)
        if (!marked[i]) {
            data_queue.push_back(data_list[i]);
        }
        else {
            if (data_list[i].metadata->num_tokens() > max_num_tokens) {
                max_num_tokens = data_list[i].metadata->num_tokens();
                max_batch_id = results.size();
            }
        }
    if (num_tokens < max_batch_size && max_batch_id != -1) {
        // TODO: consider the schedule order of this max_batch
        results.push_back(data_list[max_batch_id]);
        marked[max_batch_id] = 1;
        num_tokens += max_num_tokens;
    }
    data_list = std::move(data_queue);
    return num_tokens;
}

template<class DataBatch = AttentionBatch>
int schedule_with_limit_slice(std::vector<DataBatch> &data_list,
                              std::vector<DataBatch> &results,
                              int max_batch_size) {
    int num_tokens = 0;
    std::vector<DataBatch>().swap(results);
    std::reverse(data_list.begin(), data_list.end());
    for (int i = data_list.size() - 1; i >= 0; i --) {
        int cur_num_tokens = data_list[i].metadata->num_tokens();
        int next_num_tokens = num_tokens + cur_num_tokens;
        if (next_num_tokens <= max_batch_size) {
            results.push_back(data_list[i]);
            num_tokens = next_num_tokens;
            data_list.pop_back();
            if (num_tokens == max_batch_size)
                break;
        } else {
            int sliced_num_tokens = cur_num_tokens - (next_num_tokens - max_batch_size);
            ASSERT(0 < sliced_num_tokens && sliced_num_tokens < cur_num_tokens);
            auto pr = data_list[i].metadata->split(sliced_num_tokens);
            auto tensor_l = data_list[i].data.slice(0, 0, sliced_num_tokens);
            auto tensor_r = data_list[i].data.slice(0, sliced_num_tokens);
            ASSERT(pr.second->num_tokens() > 0);
            ASSERT(tensor_l.size(0) == pr.first->num_tokens());
            ASSERT(tensor_r.size(0) == pr.second->num_tokens());
            results.push_back((DataBatch) {tensor_l, pr.first});
            data_list[i].data = tensor_r;
            data_list[i].metadata = pr.second;
            num_tokens += sliced_num_tokens;
            ASSERT(num_tokens == max_batch_size);
            break;
        }
    }
    if (!data_list.empty())
        std::reverse(data_list.begin(), data_list.end());
    return num_tokens;
}

template<class DataBatch>
int schedule_with_limit(std::vector<DataBatch> &data_list,
                         std::vector<DataBatch> &results,
                         int max_batch_size,
                         bool allow_sliced = false,
                         bool use_dp = false) {
    /*
        DataBatch: TensorBatch or AttentionBatch
        Use DataBatch.metadata->num_tokens() to get the size of a batch
    */
    if (data_list.empty())
        return 0;
    
    if (use_dp)
        return schedule_with_limit_dp<DataBatch>(data_list, results, max_batch_size, allow_sliced);
    else if (allow_sliced)
        return schedule_with_limit_slice<DataBatch>(data_list, results, max_batch_size);
    else
        ASSERT(false);
}

std::vector<TensorBatch> MuPool::fetch_largest_batch() {
    // DMOE_LOG(INFO) << "fetching largest batch" << LEND;

    std::lock_guard<std::mutex> lock(this->batch_mutex);

    int id = this->largest_batch_layer_id_;

    if (id == -1) {
        // DMOE_LOG(INFO) << "No available batch" << LEND;
        return {};
    }
    // DMOE_LOG(INFO) << "Fetched " << id << "-th layer" << LEND;

    std::vector<TensorBatch> result;

    if (this->max_batch_size > 0) {
        int num_tokens = schedule_with_limit<TensorBatch>(
            this->data_queue[id], result, this->max_batch_size, 
            /*allow_sliced=*/ true, /*use_dp=*/ false);
        this->tokens_per_layer_[id] -= num_tokens;
    } else {
        result = std::move(this->data_queue[id]);
        this->data_queue[id].clear();
        this->tokens_per_layer_[id] = 0;
    }

    maintain_largest_batch();

    return result;
}

void MuPool::set_max_batch_size(int max_batch_size) {
    this->max_batch_size = max_batch_size;
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
    this->local_zmq_port_offset = 1;
    int num_layers = layer_ids.size();
    this->attn_data_queue = std::vector<std::vector<AttentionBatch>>(num_layers);
}

void MuAttentionPool::run() {
    pool_thread = std::thread([&]() {
        Recorder::create();
        MuPool::run();
    });

    if (device_group_ids.size() <= 1) {
        DMOE_LOG(WARNING) << "No group channel is needed in MuAttnPool::run." << LEND;
        return;
    }

    if (device_group_ids.size() > 1 && device_group_ids[0] != device_id) {
        DMOE_LOG(INFO) << "Running ATTN Worker pool (intra-group)" << LEND;
        group_threads.emplace_back(std::thread([&]() {
            Recorder::create();
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

                // DMOE_LOG(DEBUG) << "Worker AttnPool fetched result:" << meta << LEND;
                group_comm->recv((uintptr_t) tensor.data_ptr(), meta);
                // DMOE_LOG(DEBUG) << "Worker AttnPool broadcast finished" << LEND;
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
                Recorder::create();
                // recv messages from multiple dispatchers
                at::cuda::CUDAStream c10_stream = at::cuda::getCurrentCUDAStream(0);
                at::cuda::CUDAStreamGuard guard(c10_stream);

                while (!end_flag) {
                    // DMOE_LOG(DEBUG) << "AttnPool fetching metadata ..." << LEND;
                    Metadata meta;
                    c->recv_metadata(meta);
                    // DMOE_LOG(DEBUG) << "AttnPool fetched in stream " << c10_stream.stream() << " " << meta << LEND;

                    torch::Tensor tensor = torch::empty(
                        {meta.num_tokens(), meta.token_hidden_dim()}, 
                        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
                    );

                    // DMOE_LOG(DEBUG) << "AttnPool created tensor" << LEND;
                    c->recv((uintptr_t)tensor.data_ptr(), meta);
                    // DMOE_LOG(DEBUG) << "AttnPool broadcast finished" << LEND;

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
    ASSERT(meta.get() != nullptr);

    auto shape = meta->shape;
    auto dtype = meta->dtype;
    int layer_id = meta->layer_id;

    int num_tokens = meta->req_ids.size();

    int num_prefill_seqs = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;

    std::vector<int> seq_ids{};
    std::vector<int> init_prefill_lens{};
    std::vector<uint8_t> attn_dp_ranks{};

    ASSERT(meta->req_ids.size() == meta->attn_dp_ranks.size());

    for (int i = 0; i < meta->req_ids.size(); i ++) {
        if (meta->init_prefill_lens[i] != -1) {
            num_prefill_tokens ++;
            num_prefill_seqs ++;
            init_prefill_lens.emplace_back(meta->init_prefill_lens[i]);
        } else {
            num_decode_tokens ++;
        }
        // NOTE: Only considered for prefill length = 1
        seq_ids.emplace_back(meta->req_ids[i]);
        attn_dp_ranks.emplace_back(meta->attn_dp_ranks[i]);
    }

    auto attn_meta = std::make_shared<AttentionBatchMetadata> (AttentionBatchMetadata {
        layer_id, shape, dtype,
        num_prefill_seqs,
        num_prefill_tokens,
        num_decode_tokens,
        seq_ids,
        init_prefill_lens,
        {}, // expert_ids
        {}, // topk_weights
        attn_dp_ranks
    });

    return AttentionBatch {tensor, attn_meta};
}

void MuAttentionPool::process_batch(torch::Tensor tensor, metadata_t &meta, bool send_from_zmq) {
    // DMOE_LOG(INFO) << "AttnPool processing batch: " << *meta << LEND;
    if (send_from_zmq && meta->layer_id == 0 && group_comm.get() != nullptr) {
        // since only driver can have the pool, we can send the data from layer 0 to other workers here.
        // NOTE(hogura|20241110): group_comm is only used when send_from_zmq, so it should be thread-safe
        // DMOE_LOG(DEBUG) << "Broadcasting attn batch to workers" << LEND;
        group_comm->send_metadata(*meta);
        group_comm->send((uintptr_t) tensor.data_ptr(), *meta);
        // DMOE_LOG(DEBUG) << "Broadcast finished." << LEND;
    }

    int lid = this->layer_id_P2V[meta->layer_id];
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
        // DMOE_LOG(DEBUG) << "tokens_in_layer #" << lid << " " 
        //     << d.metadata->num_prefill_tokens << " " 
        //     << d.metadata->num_decode_tokens << LEND;
    }
    return num_tokens;
}

std::vector<AttentionBatch> MuAttentionPool::fetch_largest_batch(int *selected_layer_id) {
    // TODO(hogura|20240930): only considering decode first

    // DMOE_LOG(INFO) << "fetching largest batch" << LEND;
    int layer_id = -1;
    int num_tokens = 0;
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

        if (this->max_batch_size > 0) {
            num_tokens = schedule_with_limit<AttentionBatch>(
                this->attn_data_queue[layer_id], result, this->max_batch_size, 
                /*allow_sliced=*/ true, /*use_dp=*/ false);
            this->tokens_per_layer_[layer_id] -= num_tokens;
        } else {
            num_tokens = this->largest_batch_size_;
            result = std::move(this->attn_data_queue[layer_id]);
            this->attn_data_queue[layer_id].clear();
            this->tokens_per_layer_[layer_id] = 0;
        }

        maintain_largest_batch();
    }

    // DMOE_LOG(DEBUG) << "Fetched " << layer_id << " layer with #tokens=" << num_tokens << LEND;

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

std::vector<TokenTopKInfo> TokenTopKPool::fetch_ready_tokens() {
    std::vector<TokenTopKInfo> result(std::move(this->ready_tokens));
    return result;
}

void TokenTopKPool::put_batch(TensorBatch batch) {
    auto meta = batch.metadata;
    int n = meta->num_tokens();

    // DMOE_LOG(INFO) << "TokenTopKPool putting batch: " << *meta << LEND;
    
    for (int i = 0; i < n; i++) {
        int seq_id = meta->req_ids[i];
        auto it = this->pool_.find(seq_id);
        if (it == this->pool_.end()) {
            this->pool_[seq_id] = TokenTopKInfo(
                seq_id, 
                meta->init_prefill_lens[i], 
                meta->attn_dp_ranks[i],
                meta->topk_weights[i],
                batch.data[i]
            );
        } else {
            it->second.append_tensor(meta->topk_weights[i], batch.data[i]);
            if (it->second.count() == this->top_k) {
                // OPTMIZE: we can directy insert token info to scheduling queue to save one memory copy
                this->ready_tokens.emplace_back(it->second);
                // DMOE_LOG(INFO) << "ready token: " << it->second << LEND;
                this->pool_.erase(it);
            }
        }
    }
}

MuAttentionTopKPool::MuAttentionTopKPool(
    std::vector<int> layer_ids, 
    int device_id,
    std::vector<Channel_t> channels,
    std::vector<int> device_group_ids,
    Channel_t group_comm,
    int top_k
): MuAttentionPool(layer_ids, device_id, channels, device_group_ids, group_comm), top_k(top_k) {
    int num_layers = layer_ids.size();
    this->attn_token_queues = std::vector<std::vector<TokenTopKInfo>>(num_layers);
    this->topk_pools = std::vector<TokenTopKPool>{};
    for (auto &layer_id: layer_ids) {
        this->topk_pools.emplace_back(TokenTopKPool(top_k));
    }
}

void MuAttentionTopKPool::process_batch(torch::Tensor tensor, metadata_t &meta, bool send_from_zmq) {
    // DMOE_LOG(DEBUG) << "AttnTopKPool processing batch: " << *meta << LEND;
    if (send_from_zmq && meta->layer_id == 0 && group_comm.get() != nullptr) {
        // since only driver can have the pool, we can send the data from layer 0 to other workers here.
        // NOTE: group_comm is only used when send_from_zmq, so it should be thread-safe
        // DMOE_LOG(DEBUG) << "Broadcasting attn batch to workers" << LEND;
        group_comm->send_metadata(*meta);
        group_comm->send((uintptr_t) tensor.data_ptr(), *meta);
        // DMOE_LOG(DEBUG) << "Broadcast finished." << LEND;
    }

    int lid = this->layer_id_P2V[meta->layer_id];
    std::vector<TokenTopKInfo> ready_tokens{};
    int batched_tokens = 0;
    if (meta->layer_id == 0) {
        // NOTE: the first layer receives directly from tokenizer and sampler so there is no topk
        ASSERT (meta->topk_weights.size() == 0);
        ready_tokens = meta->unpack_tokens();
        batched_tokens = ready_tokens.size();
        for (int i = 0; i < batched_tokens; i++) {
            ready_tokens[i].topk_tensors.emplace_back(tensor[i]);
        }
    } else {
        this->topk_pools[lid].put_batch((TensorBatch) {tensor, meta});
        ready_tokens = this->topk_pools[lid].fetch_ready_tokens();
        batched_tokens = ready_tokens.size();
    }

    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);

        int &tokens_cur_layer = this->tokens_per_layer_[lid];
        tokens_cur_layer += batched_tokens;
        if (tokens_cur_layer > this->largest_batch_size_) {
            this->largest_batch_size_ = tokens_cur_layer;
            this->largest_batch_layer_id_ = lid;
        }
        for (auto &token: ready_tokens) {
            this->attn_token_queues[lid].emplace_back(token);
            // DMOE_LOG(INFO) << "layer_id: " << meta->layer_id << ", ready token: " << token.seq_id << LEND;
        }
    }
    // DMOE_LOG(INFO) << "largest batch size: " << this->largest_batch_size_ << LEND;
    
}

int MuAttentionTopKPool::tokens_in_layer(int lid) {
    return this->attn_token_queues[lid].size();
}

std::vector<TokenTopKInfo> MuAttentionTopKPool::schedule_with_limit() {
    std::vector<TokenTopKInfo> results{};
    int layer_id = this->largest_batch_layer_id_;
    // move tokens from queue to result
    auto &queue = this->attn_token_queues[layer_id];
    int n = this->max_batch_size;
    results.reserve(std::min(n, static_cast<int>(queue.size())));
    for (int i = 0; !queue.empty() && i < n; i++) {
        results.emplace_back(queue.back());
        queue.pop_back();
    }
    return results;
}

std::vector<AttentionBatch> MuAttentionTopKPool::fetch_largest_batch(int *selected_layer_id) {
    int layer_id = -1;
    int num_tokens = 0;
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

        if (this->max_batch_size > 0) {
            auto tokens = this->schedule_with_limit();
            num_tokens = tokens.size();
            this->tokens_per_layer_[layer_id] -= num_tokens;
            result.emplace_back(AttentionBatch::pack_tokens(this->layer_id_V2P[layer_id], tokens));
        } else {
            num_tokens = this->largest_batch_size_;
            auto tokens = std::move(this->attn_token_queues[layer_id]);
            this->attn_token_queues[layer_id].clear();
            this->tokens_per_layer_[layer_id] = 0;
            result.emplace_back(AttentionBatch::pack_tokens(this->layer_id_V2P[layer_id], tokens));
        }

        maintain_largest_batch();
    }

    // DMOE_LOG(INFO) << "Fetched " << layer_id << " layer with #tokens=" << num_tokens << LEND;

    // {
    //     std::lock_guard<std::mutex> lock(this->request_mutex);
    //     DMOE_LOG(WARNING) << "del cur_request_count:" << cur_request_count << " " << num_tokens << LEND;
    //     this->cur_request_count -= num_tokens;
    //     ASSERT(this->cur_request_count >= 0);
    // }

    if (selected_layer_id)
        *selected_layer_id = layer_id;
    return result;
}


#include <profiler.hpp>
#include <cstdlib>
std::shared_mutex Recorder::mtx = std::shared_mutex();
recorder_t Recorder::instance = std::make_shared<Recorder>(getenv("ENABLE_NVTX"));