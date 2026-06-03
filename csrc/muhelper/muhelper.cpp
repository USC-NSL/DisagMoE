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
#include <chrono>
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

#if USE_NIXL
#include "nixl_channel.h"
#include "nixl_context.h"
#endif

static double wall_time_s() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

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
    const bool tracing = this->tracing_enabled_.load(std::memory_order_relaxed); // [TRACING]
    bool blocked = false;
    double block_start_s = 0.0;
    int pending_before = 0;
    int yield_count = 0;
    while ((int)this->pending_sends.size() >= max_pending) {
        auto &pr = this->pending_sends.front();
        cudaError_t err = cudaEventQuery(pr.second);
        if (err == cudaSuccess) {
            CUDACHECK(cudaEventDestroy(pr.second));
            this->pending_sends.pop();
        } else {
            // [TRACING] record stall metadata only when tracing is on
            if (tracing && !blocked) {
                blocked = true;
                block_start_s = wall_time_s();
                pending_before = this->pending_sends.size();
            }
            if (tracing) yield_count++;
            std::this_thread::yield();
        }
    }
    // [TRACING] accumulate stall event
    if (tracing && blocked) {
        std::lock_guard<std::mutex> lock(this->stats_mutex_);
        this->pending_send_stalls_.emplace_back(
            block_start_s, wall_time_s(), pending_before, max_pending, yield_count);
    }
}

void MuDispatcher::send_batch_nonblocking(int cid, const TokenBatch &batch) {
    tx_range _{"MuDispatcher::send_batch_nonblocking"};

    this->drain_pending_sends_to(this->max_pending_sends_);

    const bool tracing = this->tracing_enabled_.load(std::memory_order_relaxed);
    const int log_num_tokens = batch.metadata ? batch.metadata->num_tokens() : 0;
    const size_t log_bytes = batch.metadata
        ? batch.metadata->num_element() * batch.metadata->get_datatype_size()
        : 0;
    const int log_layer_id = batch.metadata ? batch.metadata->layer_id : -1;

#if USE_NIXL
    if (this->channels[cid]->is_nixl()) {
        this->channels[cid]->send_batch(batch.data, *batch.metadata);

        MetadataWithPeerId packed_data;
        packed_data.peer_id = this->device_id;
        packed_data.metadata = *batch.metadata;
        auto* nixl_channel = static_cast<NixlChannel*>(this->channels[cid].get());
        packed_data.nixl_slot_id = nixl_channel->last_slot_id();
        packed_data.nixl_seq = nixl_channel->last_seq();
        auto data = cerealize_(packed_data);
        this->peer_mq[cid]->send(data.c_str(), data.size());

        if (tracing) {
            std::lock_guard<std::mutex> lock(this->stats_mutex_);
            this->send_msg_sizes_.emplace_back(
                cid, log_layer_id, log_num_tokens, log_bytes, wall_time_s(), /*transport=*/1);
        }
        return;
    }
#endif

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

    if (tracing) {
        const int transport_kind = this->channels[cid]->is_local() ? 2 : 0;
        std::lock_guard<std::mutex> lock(this->stats_mutex_);
        this->send_msg_sizes_.emplace_back(
            cid, log_layer_id, log_num_tokens, log_bytes, wall_time_s(), transport_kind);
    }
}

void MuDispatcher::_send_batch(int cid, uintptr_t buf, const BatchMetadata& meta) {
    tx_range _{"MuDispatcher::_send_batch"};
    // DMOE_LOG(WARNING) << "sending batch to channel " << cid << " current device: " << this->device_id_str << LEND;

#if USE_NIXL
    if (this->channels[cid]->is_nixl()) {
        this->channels[cid]->send_raw(buf, meta);

        MetadataWithPeerId packed_data;
        packed_data.peer_id = this->device_id;
        packed_data.metadata = meta;
        auto* nixl_channel = static_cast<NixlChannel*>(this->channels[cid].get());
        packed_data.nixl_slot_id = nixl_channel->last_slot_id();
        packed_data.nixl_seq = nixl_channel->last_seq();
        auto data = cerealize_(packed_data);
        this->peer_mq[cid]->send(data.c_str(), data.size());
        return;
    }
#endif

    // Pack peer_id and metadata into a single message
    MetadataWithPeerId packed_data;
    packed_data.peer_id = this->device_id;
    packed_data.metadata = meta;
    auto data = cerealize_(packed_data);
    this->peer_mq[cid]->send(data.c_str(), data.size());
    this->channels[cid]->send_raw(buf, meta);

    // DMOE_LOG(DEBUG) << "sent batch to channel " << cid << LEND;
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

    // DMOE_LOG(DEBUG) << "running mudispatcher@" << this->device_id << LEND;
    while (!this->end_flag) {
        // DMOE_LOG(WARNING) << "waiting for new dispatching request ..." << LEND;
        this->clean_pending_sends();
        TokenBatch batch;
        {
            // Fetch a batch from the queue, lock required (for the send_queue).
            std::unique_lock<std::mutex> lock(this->mtx);
            this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
            // DMOE_LOG(WARNING) << "Got a request !!!" << LEND;
            auto pr = this->send_queue.front();
            batch = pr.first;
            this->send_queue.pop();
        }
        // Send the batch, no lock required, since send_queue won't be changed.
        this->_send_once(batch);
    }
}

void MuDispatcher::put(TokenBatch batch, int rank) {
    std::lock_guard<std::mutex> lock(this->mtx);
    // batch.data = batch.data.clone().detach();
    this->send_queue.push(std::make_pair(batch, rank));
    this->cv.notify_one();
}

std::vector<std::tuple<double, double, int, int, int>> MuDispatcher::drain_pending_send_stall_stats() {
    std::lock_guard<std::mutex> lock(this->stats_mutex_);
    auto stats = this->pending_send_stalls_;
    this->pending_send_stalls_.clear();
    return stats;
}

std::vector<std::tuple<int, int, int, size_t, double, int>> MuDispatcher::drain_send_msg_size_stats() {
    std::lock_guard<std::mutex> lock(this->stats_mutex_);
    auto stats = std::move(this->send_msg_sizes_);
    this->send_msg_sizes_.clear();
    return stats;
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

std::vector<std::tuple<int, int, int, size_t, double, double, bool>> MuPool::drain_recv_completion_stats() {
    std::lock_guard<std::mutex> lock(this->recv_stats_mutex_);
    auto stats = this->recv_completions_;
    this->recv_completions_.clear();
    return stats;
}

void MuPool::recv_metadata(MetadataWithPeerId &packed_data, bool non_blocking) {
    // DMOE_LOG(DEBUG) << "fetching a msg ..." << LEND;
    std::vector<uint8_t> data;
    bool ok = mq->recv(data, non_blocking);
    if (!ok) {
        packed_data.peer_id = -1;
        packed_data.nixl_slot_id = -1;
        packed_data.nixl_seq = -1;
        return;
    }
    // Unpack peer_id and metadata from a single message
    decerealize_(reinterpret_cast<char*>(data.data()), data.size(), packed_data);
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

    const bool tracing = this->tracing_enabled_.load(std::memory_order_relaxed); // [TRACING]
    std::vector<MuPoolPendingRecv> pending_recvs;
    const bool has_nixl = std::any_of(
        this->channels.begin(), this->channels.end(),
        [](const Channel_t& channel) { return channel->is_nixl(); });

    uint64_t pool_iter_count = 0;
    uint64_t pool_iter_had_work = 0;
    uint64_t pool_iter_slept = 0;
    uint64_t pool_total_xfer_drained = 0;
    uint64_t pool_total_meta_recv = 0;
    uint64_t pool_total_data_ready = 0;
    auto pool_stats_last = std::chrono::steady_clock::now();

    while (!this->end_flag) {
        ++pool_iter_count;
        bool _iter_had_work = false;
#if USE_NIXL
        if (has_nixl) {
            auto& nixl_ctx = NixlContext::instance();
            if (nixl_ctx.has_pending_xfers()) {
                const int _drained = nixl_ctx.drain_pending_xfers();
                pool_total_xfer_drained += _drained;
                if (_drained > 0) _iter_had_work = true;
            }
            nixl_ctx.poll_notifs();
            if (nixl_ctx.has_ready_notifs() && !pending_metas_.empty())
            for (auto it = pending_metas_.begin(); it != pending_metas_.end();) {
                auto &pending = it->second;
                if (!nixl_ctx.consume_ready(pending.peer_id, pending.seq)) {
                    ++it;
                    continue;
                }

                const double t_data_ready_s = tracing ? wall_time_s() : 0.0;

                torch::Tensor tensor = torch::empty(
                    {pending.meta->num_tokens(), pending.meta->token_hidden_dim()},
                    torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
                );
                const size_t num_bytes = pending.meta->num_element() * pending.meta->get_datatype_size();
                cudaStream_t recv_strm = get_current_torch_stream(0);
                CUDACHECK(cudaMemcpyAsync(
                    tensor.data_ptr(),
                    reinterpret_cast<void*>(nixl_ctx.recv_slot_ptr(pending.peer_id, pending.slot_id)),
                    num_bytes,
                    cudaMemcpyDeviceToDevice,
                    recv_strm));
                cudaEvent_t ev;
                CUDACHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                CUDACHECK(cudaEventRecord(ev, recv_strm));

                const double t_d2d_issued_s = tracing ? wall_time_s() : 0.0;

                MuPoolPendingRecv recv_entry;
                recv_entry.peer_id = pending.peer_id;
                recv_entry.meta = pending.meta;
                recv_entry.tensor = tensor;
                recv_entry.event = ev;
                recv_entry.posted_ts_s = pending.posted_ts_s;
                recv_entry.nixl_slot_id = pending.slot_id;
                recv_entry.nixl_seq = pending.seq;
                recv_entry.nixl_bytes = num_bytes;
                recv_entry.t_meta_arrived_s = pending.t_meta_arrived_s;
                recv_entry.t_data_ready_s = t_data_ready_s;
                recv_entry.t_d2d_issued_s = t_d2d_issued_s;
                pending_recvs.push_back(std::move(recv_entry));
                it = pending_metas_.erase(it);
                ++pool_total_data_ready;
                _iter_had_work = true;
            }
        }
#endif

        bool should_block = pending_recvs.empty() && pending_metas_.empty();
#if USE_NIXL
        if (has_nixl) {
            should_block = false;
        }
#endif
        int drain_count = 0;
        bool got_anything = false;

        do {
            MetadataWithPeerId packed_data;
            recv_metadata(packed_data, /*non_blocking=*/ !should_block);
            should_block = false;

            if (packed_data.peer_id < 0)
                break;

            const int peer_id = packed_data.peer_id;
            batch_metadata_t meta = std::make_shared<BatchMetadata>(std::move(packed_data.metadata));

            torch::Tensor tensor = torch::empty(
                {meta->num_tokens(), meta->token_hidden_dim()},
                torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
            );

            auto *channel = this->peer_channels[peer_id].get();
            const bool is_local = channel->is_local();
            const bool is_nixl = channel->is_nixl();
            const double posted_ts_s = tracing ? wall_time_s() : 0.0; // [TRACING]

            if (is_local) {
                channel->recv_batch(tensor, *meta);
                channel->sync();
                // [TRACING] record local-recv completion
                if (tracing) {
                    const size_t num_bytes = meta->num_element() * meta->get_datatype_size();
                    std::lock_guard<std::mutex> lock(this->recv_stats_mutex_);
                    this->recv_completions_.emplace_back(
                        peer_id, meta->layer_id, meta->num_tokens(), num_bytes,
                        posted_ts_s, wall_time_s(), true);
                }
                this->process_batch(tensor, meta);
#if USE_NIXL
            } else if (is_nixl) {
                if (meta->num_element() == 0 || packed_data.nixl_slot_id < 0 || packed_data.nixl_seq < 0) {
                    this->process_batch(tensor, meta);
                } else {
                    NixlPendingMeta entry;
                    entry.peer_id = peer_id;
                    entry.slot_id = packed_data.nixl_slot_id;
                    entry.seq = packed_data.nixl_seq;
                    entry.meta = meta;
                    entry.posted_ts_s = posted_ts_s;
                    entry.t_meta_arrived_s = posted_ts_s;
                    pending_metas_[{peer_id, packed_data.nixl_seq}] = std::move(entry);
                }
#endif
            } else {
                channel->recv_batch(tensor, *meta);
                cudaEvent_t ev;
                CUDACHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                channel->record_event(ev);
                pending_recvs.push_back(MuPoolPendingRecv{peer_id, meta, tensor, ev, posted_ts_s, -1});
            }

            drain_count++;
        } while (drain_count < MU_POOL_GROUP_RECV_LIMIT);

        // 2. Poll pending NCCL recvs — process whichever completed
        for (auto it = pending_recvs.begin(); it != pending_recvs.end(); ) {
            if (cudaEventQuery(it->event) == cudaSuccess) {
                CUDACHECK(cudaEventDestroy(it->event));
                // [TRACING] record NCCL recv completion
                if (tracing) {
                    std::lock_guard<std::mutex> lock(this->recv_stats_mutex_);
                    this->recv_completions_.emplace_back(
                        it->peer_id,
                        it->meta->layer_id,
                        it->meta->num_tokens(),
                        it->meta->num_element() * it->meta->get_datatype_size(),
                        it->posted_ts_s,
                        wall_time_s(),
                        false);
                }
#if USE_NIXL
                if (it->nixl_slot_id >= 0) {
                    auto& nixl_ctx = NixlContext::instance();
                    const double t_event_signaled = tracing ? wall_time_s() : 0.0;
                    nixl_ctx.send_credit(it->peer_id, it->nixl_slot_id);
                    if (tracing && nixl_ctx.tracing_enabled()) {
                        const double t_credit_sent = wall_time_s();
                        nixl_ctx.record_recv_trace(NixlRecvTraceTuple{
                            it->peer_id,
                            it->nixl_seq,
                            it->nixl_slot_id,
                            it->nixl_bytes,
                            it->t_meta_arrived_s,
                            it->t_data_ready_s - it->t_meta_arrived_s,
                            t_event_signaled - it->t_d2d_issued_s,
                            t_credit_sent - t_event_signaled,
                            t_credit_sent - it->t_meta_arrived_s,
                        });
                    }
                }
#endif
                this->process_batch(it->tensor, it->meta);
                it = pending_recvs.erase(it);
            } else {
                ++it;
            }
        }
        if (drain_count > 0) {
            pool_total_meta_recv += drain_count;
            _iter_had_work = true;
        }
#if USE_NIXL
        if (has_nixl && !_iter_had_work) {
            std::this_thread::yield();
            ++pool_iter_slept;
        }
#endif
        if (_iter_had_work) ++pool_iter_had_work;

        if (pool_iter_count % 100000 == 0) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - pool_stats_last).count();
            DMOE_LOG(INFO) << "[POOL_STATS dev=" << this->device_id
                << "] iters=" << pool_iter_count
                << " had_work=" << pool_iter_had_work
                << " slept=" << pool_iter_slept
                << " meta_recv=" << pool_total_meta_recv
                << " data_ready=" << pool_total_data_ready
                << " xfer_drained=" << pool_total_xfer_drained
                << " pending_recvs=" << pending_recvs.size()
                << " pending_metas=" << pending_metas_.size()
                << " window_s=" << dt
                << " iters/sec=" << (100000.0 / std::max(dt, 1e-9))
                << LEND;
            pool_stats_last = now;
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

    static thread_local uint64_t put_count = 0;
    static thread_local uint64_t put_layer_sum = 0;
    static thread_local int put_last_layer = -1;
    static thread_local int put_last_largest = -1;
    ++put_count;
    put_layer_sum += layer_id;
    put_last_layer = layer_id;
    put_last_largest = this->largest_batch_layer_id_;
    if (put_count % 1000 == 0) {
        DMOE_LOG(INFO) << "[SCHED_PUT dev=" << this->device_id
            << "] count=" << put_count
            << " avg_layer=" << (put_layer_sum / put_count)
            << " last_layer=" << put_last_layer
            << " last_largest=" << put_last_largest
            << " largest_size=" << this->largest_batch_size_
            << " queue_depth_at_layer=" << this->attn_data_queue[put_last_layer].size()
            << LEND;
    }
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

    static thread_local uint64_t get_count = 0;
    static thread_local uint64_t get_empty_lbs = 0;
    static thread_local uint64_t get_oor = 0;
    static thread_local uint64_t get_empty_queue = 0;
    static thread_local uint64_t get_success = 0;
    static thread_local uint64_t get_layer_mismatch = 0;
    ++get_count;

    if (this->largest_batch_size_ == 0) {
        ++get_empty_lbs;
        if (get_count % 1000 == 0) {
            DMOE_LOG(INFO) << "[SCHED_GET dev=" << this->device_id
                << "] count=" << get_count
                << " empty_lbs=" << get_empty_lbs
                << " oor=" << get_oor
                << " empty_queue=" << get_empty_queue
                << " success=" << get_success
                << " layer_mismatch=" << get_layer_mismatch
                << " largest_lid=" << this->largest_batch_layer_id_
                << LEND;
        }
        return {};
    }

    if (layer_id < 0 || layer_id >= (int)this->attn_data_queue.size()) {
        ++get_oor;
        return {};
    }

    if (this->attn_data_queue[layer_id].empty()) {
        ++get_empty_queue;
        if (layer_id != this->largest_batch_layer_id_) {
            ++get_layer_mismatch;
        }
        if (get_count % 1000 == 0) {
            DMOE_LOG(INFO) << "[SCHED_GET dev=" << this->device_id
                << "] count=" << get_count
                << " empty_lbs=" << get_empty_lbs
                << " empty_queue=" << get_empty_queue
                << " success=" << get_success
                << " layer_mismatch=" << get_layer_mismatch
                << " req_lid=" << layer_id
                << " largest_lid=" << this->largest_batch_layer_id_
                << " largest_size=" << this->largest_batch_size_
                << LEND;
        }
        this->tokens_per_layer_[layer_id] = 0;
        this->num_batches_per_layer_[layer_id] = 0;
        maintain_largest_batch();
        return {};
    }

    this->tokens_per_layer_[layer_id] = 0;
    this->num_batches_per_layer_[layer_id] = 0;

    maintain_largest_batch();

    std::vector<TokenBatch> batches {};
    batches.swap(this->attn_data_queue[layer_id]);
    ++get_success;
    if (get_count % 1000 == 0) {
        DMOE_LOG(INFO) << "[SCHED_GET dev=" << this->device_id
            << "] count=" << get_count
            << " empty_lbs=" << get_empty_lbs
            << " empty_queue=" << get_empty_queue
            << " success=" << get_success
            << " layer_mismatch=" << get_layer_mismatch
            << " req_lid=" << layer_id
            << LEND;
    }
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

    static thread_local std::unordered_map<int, double> first_arrival_ts;
    static thread_local uint64_t topk_emit_count = 0;
    static thread_local double topk_sum_age_us = 0.0;
    static thread_local double topk_max_age_us = 0.0;
    static thread_local uint64_t topk_put_count = 0;
    static thread_local uint64_t topk_pool_size_sum = 0;

    const double now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    ++topk_put_count;
    topk_pool_size_sum += this->pool_.size();

    for (int i = 0; i < n; i++) {
        int seq_id = meta->req_ids[i];

        auto it = this->pool_.find(seq_id);
        if (it == this->pool_.end()) {
            this->pool_[seq_id] = TokenTopKInfo(
                seq_id,
                meta->init_prefill_lens[i],
                meta->attn_dp_ranks[i],
                batch.data[i]
            );
            first_arrival_ts[seq_id] = now;
        } else {
            it->second.append_tensor(batch.data[i]);
            if (it->second.count() == this->top_k) {
                this->ready_tokens.emplace_back(it->second);
                auto ts_it = first_arrival_ts.find(seq_id);
                if (ts_it != first_arrival_ts.end()) {
                    double age_us = (now - ts_it->second) * 1e6;
                    topk_sum_age_us += age_us;
                    if (age_us > topk_max_age_us) topk_max_age_us = age_us;
                    first_arrival_ts.erase(ts_it);
                }
                ++topk_emit_count;
                this->pool_.erase(it);
            }
        }
    }

    if (topk_put_count % 200 == 0) {
        DMOE_LOG(INFO) << "[TOPK_STATS top_k=" << this->top_k
            << "] put=" << topk_put_count
            << " emits=" << topk_emit_count
            << " avg_pool_size=" << (topk_pool_size_sum / topk_put_count)
            << " cur_pool_size=" << this->pool_.size()
            << " avg_age_us=" << (topk_emit_count > 0 ? topk_sum_age_us / topk_emit_count : 0.0)
            << " max_age_us=" << topk_max_age_us
            << " ready_pending=" << this->ready_tokens.size()
            << LEND;
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
