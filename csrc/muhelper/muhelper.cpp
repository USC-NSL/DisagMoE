#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <string>
#include <mutex>
#include <queue>
#include <ctime>
#include <utility>
#include <atomic>
#include <thread>
#include <pthread.h>

#include "distributed.hpp"
#include "datatypes.hpp"
#include "muhelper.h"
#include "comm.h"
#include "utils.hpp"
#include "logging.h"
#include "constants.h"
#include "cuda_utils.h"
#include "profiler.hpp"
#include "scheduler.h"
#include "layer.h"
#include "debugging.h"

#include "transport_factory.h"

#include <cereal/archives/binary.hpp>

// Struct to pack peer_id and metadata together
struct MetadataWithPeerId {
    int peer_id;
    BatchMetadata metadata;

    template<class Archive>
    void serialize(Archive &archive) {
        archive(peer_id, metadata);
    }
};

// MuHelper

MuHelper::MuHelper(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    layer_ids(layer_ids), device_id(device_id), channels(channels), end_flag(false) { }

MuHelper::~MuHelper() {}

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
#if defined(D_ENABLE_HANG_DEBUGGER) && D_ENABLE_HANG_DEBUGGER == 1
    HangDebugger::startMonThread(device_id);
#endif
}

int MuHelper::get_device_id() {
    return device_id;
}

void MuHelper::terminate() {
    this->end_flag = true;
    this->thread.join();
#if defined(D_ENABLE_HANG_DEBUGGER) && D_ENABLE_HANG_DEBUGGER == 1
    HangDebugger::terminate();
#endif
}

void MuHelper::init_cuda_device() {
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->device_id));
    #endif
}

// MuDispatcher

MuDispatcher::MuDispatcher(std::vector<int> layer_ids, int device_id, 
                           ParallelConfig cfg, std::vector<Channel_t> channels): 
    MuHelper(layer_ids, device_id, channels)
    , cfg(cfg) {
    sprintf(this->device_id_str, "%d", this->device_id);
    peer_mq.resize(channels.size());
    for (int i = 0; i < channels.size(); i ++) {
        peer_mq[i] = disagmoe::mq_factory()(/*isPush=*/ true);
    }
}

void MuDispatcher::clean_pending_sends() {
    static int spin_count = 0;
    while (!this->pending_sends.empty()) {
        auto &pr = this->pending_sends.front();
        cudaError_t err = cudaEventQuery(pr.second);
        if (err == cudaSuccess) {
            CUDACHECK(cudaEventDestroy(pr.second));
            this->pending_sends.pop();
            spin_count = 0;
        } else if (err == cudaErrorNotReady) {
            spin_count ++;
            if (spin_count > 10000) {
                DMOE_LOG(ERROR) << "spin count too large: " << spin_count << LEND;
                while (!this->pending_sends.empty()) {
                    auto &pr = this->pending_sends.front();
                    DMOE_LOG(ERROR) << "pending send: metadata=" << *pr.first.metadata << LEND;
                    this->pending_sends.pop();
                }
                ASSERT_MSG(false, "spin count too large");
            }
            break;
        } else {
            DMOE_LOG(ERROR) << "cudaEventQuery failed: " << cudaGetErrorName(err) << ", error string: " << cudaGetErrorString(err) << ", spin count: " << spin_count << LEND;
            ASSERT_MSG(false, "Failed to query cuda event");
        }
    }
}

void MuDispatcher::drain_pending_sends_to(int max_pending) {
    while ((int)this->pending_sends.size() >= max_pending) {
        auto &pr = this->pending_sends.front();
        cudaError_t err = cudaEventQuery(pr.second);
        if (err == cudaSuccess) {
            CUDACHECK(cudaEventDestroy(pr.second));
            this->pending_sends.pop();
        } else {
            std::this_thread::yield();
        }
    }
}

void MuDispatcher::send_batch_nonblocking(int cid, const TokenBatch &batch) {
    tx_range _{"MuDispatcher::send_batch_nonblocking"};

    if (xfer_config_.enabled) {
        int flat_lid = compute_flat_lid(*batch.metadata);
        _accumulate_for_xfer(cid, batch, flat_lid);
        return;
    }

    this->drain_pending_sends_to(this->max_pending_sends_);

    MetadataWithPeerId packed_data;
    packed_data.peer_id = this->device_id;
    packed_data.metadata = *batch.metadata;
    auto data = cerealize_(packed_data);
    this->peer_mq[cid]->send(data.c_str(), data.size());
    this->channels[cid]->send_batch(batch.data, *batch.metadata);

    cudaEvent_t event;
    CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    this->channels[cid]->record_event(event);
    this->pending_sends.push(std::make_pair(batch, event));
}

void MuDispatcher::_send_batch(int cid, uintptr_t buf, const BatchMetadata& meta) {
    tx_range _{"MuDispatcher::_send_batch"};

    if (xfer_config_.enabled) {
        ASSERT(current_send_tensor_.defined());
        uintptr_t base = (uintptr_t)current_send_tensor_.data_ptr();
        int hidden_bytes = meta.token_hidden_dim() * meta.get_datatype_size();
        int token_offset = (int)((buf - base) / hidden_bytes);
        int num_tokens = meta.num_tokens();

        auto tensor_view = current_send_tensor_.narrow(0, token_offset, num_tokens);
        auto meta_ptr = std::make_shared<BatchMetadata>(meta);
        int flat_lid = compute_flat_lid(meta);

        _accumulate_for_xfer(cid, TokenBatch{tensor_view, meta_ptr}, flat_lid);
        return;
    }

    MetadataWithPeerId packed_data;
    packed_data.peer_id = this->device_id;
    packed_data.metadata = meta;
    auto data = cerealize_(packed_data);
    this->peer_mq[cid]->send(data.c_str(), data.size());
    this->channels[cid]->send_raw(buf, meta);
}

void MuDispatcher::run() {
    std::string th_name = "MuDispatcher@" + std::to_string(this->device_id);
    pthread_setname_np(pthread_self(), th_name.c_str());
#if defined(D_ENABLE_HANG_DEBUGGER) && D_ENABLE_HANG_DEBUGGER == 1
    int timeout4dump = HangDebugger::calcDumpTimeout(this->device_id, 0);
    HangDebugger::registerTimeoutForStackDump(this->thread.native_handle(), timeout4dump, th_name);
#endif
    cudaDeviceSynchronize();
    const auto &make_endpoint = disagmoe::mq_endpoint_factory();
    for (int i = 0; i < this->channels.size(); i ++) {
        auto endpoint = make_endpoint(this->channels[i]->get_peer_id(), true, -1);
        this->peer_mq[i]->connect(endpoint);
    }

    while (!this->end_flag) {
        this->clean_pending_sends();

        if (!xfer_config_.enabled) {
            TokenBatch batch;
            {
                std::unique_lock<std::mutex> lock(this->mtx);
                this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
                auto pr = this->send_queue.front();
                batch = pr.first;
                this->send_queue.pop();
            }
            this->_send_once(batch);
        } else {
            std::vector<TokenBatch> to_process;
            {
                std::unique_lock<std::mutex> lock(this->mtx);
                this->cv.wait(lock, [&] {
                    return this->end_flag || !this->send_queue.empty();
                });
                while (!this->send_queue.empty()) {
                    to_process.push_back(this->send_queue.front().first);
                    this->send_queue.pop();
                }
            }

            if (to_process.empty()) continue;

            for (auto& batch : to_process) {
                this->current_send_tensor_ = batch.data;
                this->_send_once(batch);
            }
            this->current_send_tensor_ = torch::Tensor();

            this->_flush_xfer_buffers();
        }
    }

    if (xfer_config_.enabled) {
        int saved = xfer_config_.max_channels_per_cycle;
        xfer_config_.max_channels_per_cycle = (int)xfer_buffers_.size();
        this->_flush_xfer_buffers();
        xfer_config_.max_channels_per_cycle = saved;
    }
}

void MuDispatcher::put(TokenBatch batch, int rank) {
    std::lock_guard<std::mutex> lock(this->mtx);
    this->send_queue.push(std::make_pair(batch, rank));
    this->cv.notify_one();
}

void MuDispatcher::terminate() {
    {
        std::lock_guard<std::mutex> lock(this->mtx);
        this->end_flag = true;
    }
    this->cv.notify_one();
    this->thread.join();
#if defined(D_ENABLE_HANG_DEBUGGER) && D_ENABLE_HANG_DEBUGGER == 1
    HangDebugger::terminate();
#endif
}

void MuDispatcher::set_xfer_buffer_config(bool enabled, int max_channels, int num_layers) {
    xfer_config_.enabled = enabled;
    xfer_config_.max_channels_per_cycle = max_channels;
    xfer_config_.num_layers = num_layers;
    xfer_buffers_.resize(this->channels.size());
}

bool MuDispatcher::is_before(int a_flat, int b_flat, int num_layers) {
    int N = 2 * num_layers;
    int diff = ((b_flat - a_flat) % N + N) % N;
    return diff < num_layers;
}

int MuDispatcher::compute_flat_lid(const BatchMetadata& meta) const {
    return 2 * meta.layer_id + (is_expert_dispatcher_ ? 1 : 0);
}

void MuDispatcher::_accumulate_for_xfer(int cid, TokenBatch batch, int flat_lid) {
    auto& buf = xfer_buffers_[cid];
    if (buf.empty()) {
        buf.earliest_flat_lid = flat_lid;
    } else if (is_before(flat_lid, buf.earliest_flat_lid, xfer_config_.num_layers)) {
        buf.earliest_flat_lid = flat_lid;
    }
    buf.total_tokens += batch.metadata->num_tokens();
    XferBufferEntry entry;
    entry.batch = std::move(batch);
    entry.flat_lid = flat_lid;
    buf.entries.push_back(std::move(entry));
}

void MuDispatcher::_do_send_batch(int cid, const TokenBatch& batch) {
    tx_range _{"MuDispatcher::_do_send_batch"};

    this->drain_pending_sends_to(this->max_pending_sends_);

    MetadataWithPeerId packed_data;
    packed_data.peer_id = this->device_id;
    packed_data.metadata = *batch.metadata;
    auto data = cerealize_(packed_data);
    this->peer_mq[cid]->send(data.c_str(), data.size());
    this->channels[cid]->send_raw((uintptr_t)batch.data.data_ptr(), *batch.metadata);

    cudaEvent_t event;
    CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    this->channels[cid]->record_event(event);
    this->pending_sends.push(std::make_pair(batch, event));
}

void MuDispatcher::_flush_xfer_buffers() {
    tx_range _{"MuDispatcher::_flush_xfer_buffers"};

    std::vector<int> candidates;
    for (int i = 0; i < (int)xfer_buffers_.size(); i++) {
        if (!xfer_buffers_[i].empty()) {
            candidates.push_back(i);
        }
    }

    if (candidates.empty()) return;

    int num_to_send = std::min((int)candidates.size(), xfer_config_.max_channels_per_cycle);

    if ((int)candidates.size() > num_to_send) {
        int N = 2 * xfer_config_.num_layers;

        int ref = xfer_buffers_[candidates[0]].earliest_flat_lid;
        for (int cid : candidates) {
            if (is_before(xfer_buffers_[cid].earliest_flat_lid, ref, xfer_config_.num_layers)) {
                ref = xfer_buffers_[cid].earliest_flat_lid;
            }
        }

        std::partial_sort(candidates.begin(), candidates.begin() + num_to_send, candidates.end(),
            [&](int a, int b) {
                auto& ba = xfer_buffers_[a];
                auto& bb = xfer_buffers_[b];
                int dist_a = ((ba.earliest_flat_lid - ref) % N + N) % N;
                int dist_b = ((bb.earliest_flat_lid - ref) % N + N) % N;
                if (dist_a != dist_b) return dist_a < dist_b;
                return ba.total_tokens > bb.total_tokens;
            }
        );
    }

    for (int i = 0; i < num_to_send; i++) {
        int cid = candidates[i];
        auto& buf = xfer_buffers_[cid];

        std::unordered_map<int, std::vector<int>> by_layer;
        for (int e = 0; e < (int)buf.entries.size(); e++) {
            by_layer[buf.entries[e].batch.metadata->layer_id].push_back(e);
        }

        for (auto& [lid, entry_indices] : by_layer) {
            if (entry_indices.size() == 1) {
                _do_send_batch(cid, buf.entries[entry_indices[0]].batch);
            } else {
                std::vector<torch::Tensor> tensors;
                BatchMetadata merged_meta = *buf.entries[entry_indices[0]].batch.metadata;

                for (int idx : entry_indices) {
                    tensors.push_back(buf.entries[idx].batch.data);
                }

                for (size_t k = 1; k < entry_indices.size(); k++) {
                    auto& m = *buf.entries[entry_indices[k]].batch.metadata;
                    merged_meta.req_ids.insert(merged_meta.req_ids.end(),
                                               m.req_ids.begin(), m.req_ids.end());
                    merged_meta.exp_ids.insert(merged_meta.exp_ids.end(),
                                               m.exp_ids.begin(), m.exp_ids.end());
                    merged_meta.topk_weights.insert(merged_meta.topk_weights.end(),
                                                    m.topk_weights.begin(), m.topk_weights.end());
                    merged_meta.attn_dp_ranks.insert(merged_meta.attn_dp_ranks.end(),
                                                     m.attn_dp_ranks.begin(), m.attn_dp_ranks.end());
                    merged_meta.init_prefill_lens.insert(merged_meta.init_prefill_lens.end(),
                                                         m.init_prefill_lens.begin(), m.init_prefill_lens.end());
                }

                merged_meta.shape[0] = 0;
                for (auto& t : tensors) merged_meta.shape[0] += t.size(0);

                torch::Tensor merged_tensor = torch::cat(tensors, 0);
                auto merged_meta_ptr = std::make_shared<BatchMetadata>(std::move(merged_meta));
                _do_send_batch(cid, TokenBatch{merged_tensor, merged_meta_ptr});
            }
        }

        buf.clear();
    }
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
    // DMOE_LOG(INFO) << "max_layer_id " << max_layer_id << ", max_exp_id " << max_exp_id << LEND;
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
        if (out_channel_infos[i].expert_ids.empty()) {
            continue;
        }
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

void MuAttnDispatcher::_send_once(TokenBatch batch) {
    tx_range _{"MuAttnDispatcher::_send_once"};
    // DMOE_LOG(INFO) << "attn " << this->device_id << " sending a batch: " << *batch.metadata << LEND;
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
            return;
        }

        auto sliced_meta = batch.metadata->slice(i, j);

        auto buf = tensor_at((uintptr_t)batch.data.data_ptr(), batch.metadata, i);
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
    std::vector<ChannelInfo> channel_infos): 
        MuDispatcher(layer_ids, device_id, cfg, channels),
        channel_infos(channel_infos) {
    this->is_expert_dispatcher_ = true;
    int max_layer = -1;
    for (auto info: channel_infos)
        for (int i: info.attn_layer_ids)
            max_layer = std::max(i, max_layer);

    // attn_channel[layer_id][dp_rank]
    this->attn_channel.resize(max_layer + 1, {});
    for (int i = 0; i <= max_layer; i ++)
        this->attn_channel[i].resize(cfg.dp, -1);

    for (size_t i = 0; i < channels.size(); i ++) {
        ASSERT (!channel_infos[i].attn_layer_ids.empty());
        int dp_rank = channel_infos[i].attn_dp_rank;
        for (int j = 0; j < channel_infos[i].attn_layer_ids.size(); j ++) {
            int lid = channel_infos[i].attn_layer_ids[j];
            // DMOE_LOG(DEBUG) << "channel " << i << " attn_layer_id " << lid << " dp_rank " << dp_rank << LEND;
            ASSERT(this->attn_channel[lid][dp_rank] == -1);
            this->attn_channel[lid][dp_rank] = i;
        }
    }

    // DMOE_LOG(INFO) << "inited MuExpertDispatcher " << device_id << LEND;
}

int MuExpertDispatcher::_get_attn_channel(int layer_id, int rank) {
    // DMOE_LOG(DEBUG) << "layer_id: " << layer_id << " attn_chan.size: " << attn_channel.size() << LEND;
    return layer_id < this->attn_channel.size() ? this->attn_channel[layer_id][rank] : this->attn_channel[0][rank];
}

void MuExpertDispatcher::debug_put(TokenBatch batch) {
    _send_once(batch);
}

void MuExpertDispatcher::_send_once(TokenBatch batch) {
    tx_range _{"MuExpertDispatcher::_send_once"};
    auto meta = batch.metadata;
    auto layer_id = meta->layer_id;

    // DMOE_LOG(INFO) << "expert " << device_id << " sending a batch: " << *meta << ", n_ele=" << batch.data.numel()  << LEND;
    ASSERT(batch.data.sizes()[0] == meta->shape[0]);
    ASSERT(batch.data.sizes()[1] == meta->shape[1]);

    auto &channels = this->attn_channel[layer_id];

    // ncclGroupStart/End removed — see MuAttnDispatcher::_send_once for rationale.
    for (int i = 0, j = 1, n = meta->attn_dp_ranks.size(); i < n; i = j) {
        int rank = meta->attn_dp_ranks[i];
        auto channel_id = this->_get_attn_channel(layer_id, rank);
        ASSERT(0 <= rank && rank < channels.size());
        while (j < n && meta->attn_dp_ranks[j] == rank)
            j ++;

        if (i == 0 && j == n) {
            this->_send_batch(
                channel_id,
                (uintptr_t) batch.data.data_ptr(),
                *meta
            );
            return;
        } else {
            auto buf = tensor_at((uintptr_t) batch.data.data_ptr(), batch.metadata, i);
            this->_send_batch(
                channel_id,
                buf,
                batch.metadata->slice(i, j)
            );
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
    int num_groups
):  MuHelper(layer_ids, device_id, channels),
    num_groups(num_groups), 
    mq(disagmoe::mq_factory()(/*isPush=*/ false)) {
    int num_layers = layer_ids.size();
    int max_layer_id = 0;
    for (auto id: layer_ids)
        max_layer_id = std::max(max_layer_id, id);
    this->num_layers = num_layers;
    this->layer_id_P2V = std::vector<int>(max_layer_id + 1);
    this->layer_id_V2P = std::vector<int>(num_layers);

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

    this->tokens_per_layer_ = std::vector<int>(num_layers * num_groups, 0);
    this->num_batches_per_layer_ = std::vector<int>(num_layers * num_groups, 0);
    this->queueing_timers = std::map<int, clock_t>();
}

MuPool::~MuPool() {}

void MuPool::recv_metadata(int &peer_id, batch_metadata_t &meta, bool non_blocking) {
    // DMOE_LOG(DEBUG) << "fetching a msg ..." << LEND;
    std::vector<uint8_t> data;
    bool ok = mq->recv(data, non_blocking);
    if (!ok) {
        meta = nullptr;
        return;
    }
    // Unpack peer_id and metadata from a single message
    MetadataWithPeerId packed_data;
    decerealize_(reinterpret_cast<char*>(data.data()), data.size(), packed_data);
    peer_id = packed_data.peer_id;
    meta = std::make_shared<BatchMetadata>(std::move(packed_data.metadata));
    // DMOE_LOG(INFO) << "receive metadata: " << *meta << LEND;
}

void MuPool::put_batch(TokenBatch batch) {
    // CAREFUL USE:
    // This is only used to directly put a batch into the first attention layer.
    batch.data = batch.data.clone().detach();
    batch.metadata->batch_tag = BatchTag::TOKENIZER;
    this->process_batch(batch.data, batch.metadata);
}

void MuPool::start_queueing_timer(const std::vector<int> &req_ids) {
    if (req_ids.empty())
        return;
    
    std::lock_guard<std::mutex> lock(this->timer_mutex);
    for (int req_id: req_ids) {
        if (this->queueing_timers.find(req_id) == this->queueing_timers.end())
            this->queueing_timers[req_id] = t_now();
        else {
            ASSERT(this->queueing_timers.at(req_id) == -1);
            this->queueing_timers.erase(req_id);
        }
    }
}

float MuPool::remove_queueing_timer(const std::vector<int> &req_ids) {
    if (req_ids.empty())
        return 0;

    std::lock_guard<std::mutex> lock(this->timer_mutex);
    float total_delay = 0;
    auto now = t_now();
    for (int req_id: req_ids) {
        if (this->queueing_timers.find(req_id) == this->queueing_timers.end()) {
            this->queueing_timers[req_id] = -1;
            continue;
        }
        // t_now() now returns microseconds since an epoch; convert to seconds
        total_delay += 1.0 * (now - this->queueing_timers.at(req_id)) / 1e6;
        this->queueing_timers.erase(req_id);
    }
    return total_delay / req_ids.size();
}

void MuPool::run() {
    std::string th_name = "MuPool@" + std::to_string(this->device_id);
    pthread_setname_np(pthread_self(), th_name.c_str());
#if defined(D_ENABLE_HANG_DEBUGGER) && D_ENABLE_HANG_DEBUGGER == 1
    int timeout4dump = HangDebugger::calcDumpTimeout(this->device_id, 2);
    HangDebugger::registerTimeoutForStackDump(this->thread.native_handle(), timeout4dump, th_name);
#endif
    cudaDeviceSynchronize();
    if (this->channels.empty()) {
        DMOE_LOG(WARNING) << this->device_id << " has no channels, exit MuPool." << LEND;
        return;
    }
    auto pool_endpoint = disagmoe::mq_endpoint_factory()(this->device_id, true, -1);
    this->mq->bind(pool_endpoint);

    std::vector<MuPoolPendingRecv> pending_recvs;

    while (!this->end_flag) {
        // 1. Receive metadata: block only if nothing is pending, else non-blocking
        bool should_block = pending_recvs.empty();
        int drain_count = 0;

        do {
            int peer_id;
            batch_metadata_t meta;
            recv_metadata(peer_id, meta, /*non_blocking=*/ !should_block);
            should_block = false;

            if (meta.get() == nullptr)
                break;

            torch::Tensor tensor = torch::empty(
                {meta->num_tokens(), meta->token_hidden_dim()},
                torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
            );

            auto *channel = this->peer_channels[peer_id].get();
            bool is_local = (dynamic_cast<NcclChannel*>(channel) == nullptr);

            if (is_local) {
                // TensorLocalChannel: synchronous, completes immediately
                channel->recv_batch(tensor, *meta);
                channel->sync();
                this->process_batch(tensor, meta);
            } else {
                // NcclChannel: post recv, record event, add to pending queue
                channel->recv_batch(tensor, *meta);
                cudaEvent_t ev;
                CUDACHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                channel->record_event(ev);
                pending_recvs.push_back(MuPoolPendingRecv{peer_id, meta, tensor, ev});
            }

            drain_count++;
        } while (drain_count < MU_POOL_GROUP_RECV_LIMIT);

        // 2. Poll pending NCCL recvs — process whichever completed
        for (auto it = pending_recvs.begin(); it != pending_recvs.end(); ) {
            if (cudaEventQuery(it->event) == cudaSuccess) {
                CUDACHECK(cudaEventDestroy(it->event));
                this->process_batch(it->tensor, it->meta);
                it = pending_recvs.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Cleanup any remaining pending recvs
    for (auto &p : pending_recvs) {
        cudaEventDestroy(p.event);
    }
}

// the batch_mutex must be used outside this function
int MuPool::tokens_in_layer(int lid) {
    return this->tokens_per_layer_[lid];
}

int MuPool::num_batches_in_layer(int lid) {
    return this->num_batches_per_layer_[lid];
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
    // if (num_groups == 1) {
    //     return this->tokens_per_layer_;
    // }
    // std::vector<int> snapshot(this->num_layers, 0);
    // for (int i = 0; i < this->num_layers; i++) {
    //     for (int j = 0; j < this->num_groups; j++)
    //         snapshot[i] += this->tokens_per_layer_[get_layer_group_id(i, j)];
    // }
    // return snapshot;
}

MuExpertPool::MuExpertPool(
    std::vector<int> layer_ids,
    int device_id,
    std::vector<Channel_t> channels,
    int num_groups):
    MuPool(layer_ids, device_id, channels, num_groups) {
    int num_layers = layer_ids.size();
    this->data_queue = std::vector<std::vector<TokenBatch>>(num_layers * num_groups);
}

void MuExpertPool::process_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    meta->batch_tag = BatchTag::EXPERT;
    int layer_id = this->layer_id_P2V[meta->layer_id];

    auto add_one_batch = [&](int qid, const TokenBatch &batch) {
        // NOTE: batch_mutex should be held outside this function
        int num_tokens = batch.metadata->num_tokens();
        this->data_queue[qid].push_back(batch);
        this->layer_scheduler->add_tokens_to_layer(qid, num_tokens);
        this->num_batches_per_layer_[qid] += 1;
        int &tokens_cur_layer = this->tokens_per_layer_[qid];
        tokens_cur_layer += num_tokens;
        if (tokens_cur_layer > this->largest_batch_size_) {
            this->largest_batch_size_ = tokens_cur_layer;
            this->largest_batch_layer_id_ = qid;
        }
    };

    if (this->num_groups > 1) {
        TokenBatch recv_batch = TokenBatch{tensor, meta};
        std::vector<TokenBatch> batches = recv_batch.split_by_expert();
        std::lock_guard<std::mutex> lock(this->batch_mutex);
        for (auto &batch: batches) {
            int expert_id = batch.metadata->get_expert_id();
            int qid = get_layer_group_id(layer_id, expert_id % this->num_groups);
            add_one_batch(qid, batch);
        }
    } else {
        std::lock_guard<std::mutex> lock(this->batch_mutex);
        add_one_batch(layer_id, TokenBatch{tensor, meta});
    }
}

TokenBatch MuExpertPool::get_batch_from_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(this->batch_mutex);

    if (this->largest_batch_size_ == 0) {
        return TokenBatch {};
    }

    if (layer_id < 0 || layer_id >= (int)this->data_queue.size()) {
        return TokenBatch {};
    }

    if (this->tokens_per_layer_[layer_id] == 0 || this->data_queue[layer_id].empty()) {
        return TokenBatch {};
    }

    this->tokens_per_layer_[layer_id] = 0;
    this->num_batches_per_layer_[layer_id] = 0;

    maintain_largest_batch();
    
    std::vector<TokenBatch> batches {};
    batches.swap(this->data_queue[layer_id]);
    return TokenBatch::merge(batches);
}

// void MuPool::set_scheduler_block(int step) {
//     this->layer_scheduler->set_block_step(step);
// }

// void MuPool::set_layer_schedule_type(std::string type) {
//     this->layer_scheduler->set_schedule_type(type);
// }

MuAttentionPool::MuAttentionPool(
    std::vector<int> layer_ids, 
    int device_id,
    std::vector<Channel_t> channels
):  MuPool([&]() {
        layer_ids.emplace_back(layer_ids.back() + 1);
        return layer_ids;
    }(), device_id, channels, /* num_groups */ 1) {
    int num_layers = layer_ids.size();
    this->attn_data_queue = std::vector<std::vector<TokenBatch>>(num_layers);
}

TokenBatch MuAttentionPool::pack_attn_batch(torch::Tensor tensor, batch_metadata_t meta) {
    ASSERT(meta.get() != nullptr);
    int num_prefill_seqs = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;

    for (int i = 0; i < meta->req_ids.size(); i ++) {
        if (meta->init_prefill_lens[i] != -1) {
            num_prefill_tokens ++;
            num_prefill_seqs ++;
        } else {
            num_decode_tokens ++;
        }
    }

    meta->num_prefill_seqs = num_prefill_seqs;
    meta->num_prefill_tokens = num_prefill_tokens;
    meta->num_decode_tokens = num_decode_tokens;
    return TokenBatch {tensor, meta};
}

void MuAttentionPool::put_batch_to_attn_queue(int layer_id, const TokenBatch &attn_batch) {
    std::lock_guard<std::mutex> lock(this->batch_mutex);
    int batched_tokens = attn_batch.metadata->num_decode_tokens.value() + attn_batch.metadata->num_prefill_tokens.value();
    this->num_batches_per_layer_[layer_id] += 1;
    this->layer_scheduler->add_tokens_to_layer(layer_id, batched_tokens);
    int &tokens_cur_layer = this->tokens_per_layer_[layer_id];
    tokens_cur_layer += batched_tokens;
    if (tokens_cur_layer > this->largest_batch_size_) {
        this->largest_batch_size_ = tokens_cur_layer;
        this->largest_batch_layer_id_ = layer_id;
    }

    this->attn_data_queue[layer_id].push_back(attn_batch);
}

void MuAttentionPool::process_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    // DMOE_LOG(INFO) << "AttnPool processing batch: " << *meta << LEND;
    meta->batch_tag = BatchTag::ATTENTION;
    int lid = this->layer_id_P2V[meta->layer_id];
    auto attn_batch = pack_attn_batch(tensor, meta);
    this->put_batch_to_attn_queue(lid, attn_batch);
}

TokenBatch MuAttentionPool::get_batch_from_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(this->batch_mutex);

    if (this->largest_batch_size_ == 0) {
        return {};
    }

    if (layer_id < 0 || layer_id >= (int)this->attn_data_queue.size()) {
        return {};
    }

    this->tokens_per_layer_[layer_id] = 0;
    this->num_batches_per_layer_[layer_id] = 0;

    maintain_largest_batch();

    std::vector<TokenBatch> batches {};
    batches.swap(this->attn_data_queue[layer_id]);
    return TokenBatch::merge(batches);
}

std::vector<TokenTopKInfo> TokenTopKPool::fetch_ready_tokens() {
    std::vector<TokenTopKInfo> result{};
    result.swap(this->ready_tokens);
    return result;
}

void TokenTopKPool::put_batch(TokenBatch batch) {
    auto meta = batch.metadata;
    ASSERT_MSG(meta.get() != nullptr, "Metadata is nullptr");
    ASSERT_MSG(batch.data.sizes()[0] == meta->num_tokens(), "Batch data shape mismatch");
    ASSERT_MSG(batch.data.sizes()[1] == meta->token_hidden_dim(), "Batch data shape mismatch");
    ASSERT_MSG(meta->num_tokens() == meta->req_ids.size(), "Batch data shape mismatch");
    ASSERT_MSG(meta->num_tokens() == meta->attn_dp_ranks.size(), "Batch data shape mismatch");
    ASSERT_MSG(meta->num_tokens() == meta->init_prefill_lens.size(), "Batch data shape mismatch");

    int n = meta->num_tokens();

    // DMOE_LOG(INFO) << "TokenTopKPool putting batch: " << *meta << LEND;
    
    for (int i = 0; i < n; i++) {
        int seq_id = meta->req_ids[i];
        // DMOE_LOG(INFO) << "seq " << seq_id << ", dp rank " << meta->attn_dp_ranks[i] << LEND;

        auto it = this->pool_.find(seq_id);
        if (it == this->pool_.end()) {
            this->pool_[seq_id] = TokenTopKInfo(
                seq_id, 
                meta->init_prefill_lens[i], 
                meta->attn_dp_ranks[i],
                batch.data[i]
            );
        } else {
            it->second.append_tensor(batch.data[i]);
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
    int top_k
): MuAttentionPool(layer_ids, device_id, channels), top_k(top_k) {
    int num_layers = layer_ids.size();
    this->attn_token_queues = std::vector<std::vector<TokenTopKInfo>>(num_layers);
    this->topk_pools = std::vector<TokenTopKPool>{};
    for (int i = 0; i < this->num_layers; i++) {
        this->topk_pools.emplace_back(TokenTopKPool(top_k));
    }
}

void MuAttentionTopKPool::process_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    // DMOE_LOG(DEBUG) << "AttnTopKPool processing batch: " << *meta << LEND;
    meta->batch_tag = BatchTag::ATTENTION;
    int lid = this->layer_id_P2V[meta->layer_id];
    std::vector<TokenTopKInfo> ready_tokens{};
    int batched_tokens = 0;
    if (meta->layer_id == 0) {
        auto attn_batch = this->pack_attn_batch(tensor, meta);
        this->put_batch_to_attn_queue(lid, attn_batch);
        return;
    } 

    this->topk_pools[lid].put_batch((TokenBatch) {tensor, meta});
    ready_tokens = this->topk_pools[lid].fetch_ready_tokens();
    batched_tokens = ready_tokens.size();

    if (batched_tokens == 0) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(this->batch_mutex);
        this->num_batches_per_layer_[lid] += 1;
        int &tokens_cur_layer = this->tokens_per_layer_[lid];
        tokens_cur_layer += batched_tokens;
        this->layer_scheduler->add_tokens_to_layer(lid, batched_tokens);
        if (tokens_cur_layer > this->largest_batch_size_) {
            this->largest_batch_size_ = tokens_cur_layer;
            this->largest_batch_layer_id_ = lid;
        }
        for (auto &token: ready_tokens) {
            this->attn_token_queues[lid].emplace_back(token);
            // DMOE_LOG(INFO) << "layer_id: " << meta->layer_id << ", ready token: " << token.seq_id << ", dp rank: " << token.attn_dp_rank << LEND;
        }
    }
    // DMOE_LOG(INFO) << "largest batch size: " << this->largest_batch_size_ << LEND;
    
}

int MuAttentionTopKPool::tokens_in_layer(int lid) {
    return this->attn_token_queues[lid].size();
}


TokenBatch MuAttentionTopKPool::get_batch_from_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(this->batch_mutex);

    if (this->largest_batch_size_ == 0) {
        return TokenBatch {};
    }

    if (layer_id < 0 || layer_id >= (int)this->attn_token_queues.size()) {
        return TokenBatch {};
    }

    this->tokens_per_layer_[layer_id] = 0;
    this->num_batches_per_layer_[layer_id] = 0;

    maintain_largest_batch();

    auto batch = TokenBatch::pack_topk_tokens(this->layer_id_V2P[layer_id], this->attn_token_queues[layer_id]);
    this->attn_token_queues[layer_id].clear();
    return batch;
}

#include <profiler.hpp>
#include <cstdlib>
std::shared_mutex Recorder::mtx = std::shared_mutex();
recorder_t Recorder::instance = std::make_shared<Recorder>(getenv("ENABLE_NVTX"));
