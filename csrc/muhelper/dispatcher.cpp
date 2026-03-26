#include "dispatcher.h"
#include "utils.hpp"
#include "logging.h"
#include "cuda_utils.h"
#include "profiler.hpp"
#include "debugging.h"
#include "transport_factory.h"

#include <cereal/archives/binary.hpp>
#include <chrono>
#include <pthread.h>

struct MetadataWithPeerId {
    int peer_id;
    BatchMetadata metadata;

    template<class Archive>
    void serialize(Archive &archive) {
        archive(peer_id, metadata);
    }
};

// ── helpers ────────────────────────────────────────────────────────────

int UnifiedDispatcher::_attn_get_channel_id(int dp_rank) {
    return this->rank_to_channel[dp_rank];
}

int UnifiedDispatcher::_expert_get_channel_id(int expert_id) {
    int rank = this->expert_to_rank[expert_id];
    return this->rank_to_channel[rank];
}

// ── constructor ────────────────────────────────────────────────────────

UnifiedDispatcher::UnifiedDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    ParallelConfig cfg,
    std::vector<Channel_t> channels,
    std::vector<ChannelInfo> channel_infos
): MuDispatcher(layer_ids, device_id, cfg, channels) {
    int num_experts = cfg.n_total_experts > 0 ? cfg.n_total_experts
                                              : cfg.ep * cfg.n_exp_per_rank;
    this->rank_to_channel.resize(cfg.dp, -1);
    this->expert_to_rank.resize(num_experts, -1);

    for (auto &tuple: cfg.expert_ranks) {
        int exp_id = std::get<1>(tuple);
        int rank   = std::get<2>(tuple);
        this->expert_to_rank[exp_id] = rank;
    }

    for (size_t i = 0; i < channels.size(); i++) {
        int dp_rank = channel_infos[i].attn_dp_rank;
        this->rank_to_channel[dp_rank] = i;
    }

    rank_queues_.resize(cfg.dp);
    for (int r = 0; r < cfg.dp; r++) {
        rank_queues_[r].channel_id = rank_to_channel[r];
    }
}

// ── _send_once  (virtual from MuDispatcher) ────────────────────────────

void UnifiedDispatcher::_send_once(TokenBatch batch) {
    if (batch.metadata->is_attention()) {
        this->_send_to_expert_once(batch);
    } else if (batch.metadata->is_expert()) {
        this->_send_to_attn_once(batch);
    }
}

// ── split-by-rank → enqueue ────────────────────────────────────────────

void UnifiedDispatcher::_send_to_expert_once(TokenBatch batch) {
    tx_range _{"UnifiedDispatcher::_send_to_expert_once"};

    std::vector<int> split_sizes;
    std::vector<int> dest_ranks;
    for (int i = 0, j = 1, n = batch.metadata->exp_ids.size(); i < n; i = j) {
        int rank = this->expert_to_rank[batch.metadata->exp_ids[i]];
        while (j < n && this->expert_to_rank[batch.metadata->exp_ids[j]] == rank)
            j++;
        split_sizes.push_back(j - i);
        dest_ranks.push_back(rank);
    }

    auto sub_batches = batch.split_with_sizes(split_sizes);
    for (size_t k = 0; k < sub_batches.size(); k++) {
        _enqueue_for_rank(dest_ranks[k], sub_batches[k]);
    }
}

void UnifiedDispatcher::_send_to_attn_once(TokenBatch batch) {
    tx_range _{"UnifiedDispatcher::_send_to_attn_once"};

    std::vector<int> split_sizes;
    std::vector<int> dest_ranks;
    for (int i = 0, j = 1, n = batch.metadata->attn_dp_ranks.size(); i < n; i = j) {
        int rank = batch.metadata->attn_dp_ranks[i];
        while (j < n && batch.metadata->attn_dp_ranks[j] == rank)
            j++;
        split_sizes.push_back(j - i);
        dest_ranks.push_back(rank);
    }

    auto sub_batches = batch.split_with_sizes(split_sizes);
    for (size_t k = 0; k < sub_batches.size(); k++) {
        _enqueue_for_rank(dest_ranks[k], sub_batches[k]);
    }
}

// ── per-rank buffering ─────────────────────────────────────────────────

void UnifiedDispatcher::_enqueue_for_rank(int dest_rank, TokenBatch batch) {
    rank_queues_[dest_rank].buffered.push_back(std::move(batch));
}

bool UnifiedDispatcher::_has_buffered_sends() const {
    for (auto& q : rank_queues_) {
        if (!q.buffered.empty()) return true;
    }
    return false;
}

// ── merge buffered batches for one rank ────────────────────────────────

TokenBatch UnifiedDispatcher::_merge_for_rank(std::vector<TokenBatch>& batches) {
    if (batches.size() == 1) return std::move(batches[0]);

    bool heterogeneous = false;
    for (size_t i = 1; i < batches.size(); i++) {
        if (batches[i].metadata->layer_id != batches[0].metadata->layer_id ||
            batches[i].metadata->batch_tag != batches[0].metadata->batch_tag) {
            heterogeneous = true;
            break;
        }
    }

    std::vector<BatchSegment> segments;
    if (heterogeneous) {
        for (auto& b : batches) {
            segments.push_back({
                b.metadata->layer_id,
                b.metadata->batch_tag,
                b.metadata->num_tokens()
            });
        }
    }

    std::vector<torch::Tensor> tensors;
    for (auto& b : batches) tensors.push_back(b.data);
    torch::Tensor merged_tensor = torch::cat(tensors, 0);

    auto merged_meta = std::make_shared<BatchMetadata>();
    merged_meta->batch_tag  = batches[0].metadata->batch_tag;
    merged_meta->layer_id   = batches[0].metadata->layer_id;
    merged_meta->dtype      = batches[0].metadata->dtype;
    merged_meta->shape      = {(size_t)merged_tensor.size(0), batches[0].metadata->shape[1]};
    merged_meta->segments   = std::move(segments);

    for (auto& b : batches) {
        auto& m = *b.metadata;
        merged_meta->req_ids.insert(merged_meta->req_ids.end(),
                                    m.req_ids.begin(), m.req_ids.end());
        merged_meta->exp_ids.insert(merged_meta->exp_ids.end(),
                                    m.exp_ids.begin(), m.exp_ids.end());
        merged_meta->topk_weights.insert(merged_meta->topk_weights.end(),
                                         m.topk_weights.begin(), m.topk_weights.end());
        merged_meta->attn_dp_ranks.insert(merged_meta->attn_dp_ranks.end(),
                                          m.attn_dp_ranks.begin(), m.attn_dp_ranks.end());
        merged_meta->init_prefill_lens.insert(merged_meta->init_prefill_lens.end(),
                                              m.init_prefill_lens.begin(), m.init_prefill_lens.end());
    }

    return TokenBatch{merged_tensor, merged_meta};
}

// ── actual NCCL+ZMQ send ───────────────────────────────────────────────

void UnifiedDispatcher::_do_rank_send(int dest_rank, const TokenBatch& batch) {
    tx_range _{"UnifiedDispatcher::_do_rank_send"};

    int cid = rank_queues_[dest_rank].channel_id;
    ASSERT(cid >= 0);

    MetadataWithPeerId packed_data;
    packed_data.peer_id  = this->device_id;
    packed_data.metadata = *batch.metadata;
    auto data = cerealize_(packed_data);
    this->peer_mq[cid]->send(data.c_str(), data.size());
    this->channels[cid]->send_batch(batch.data, *batch.metadata);

    cudaEvent_t event;
    CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    this->channels[cid]->record_event(event);

    rank_pending_sends_.push(PendingRankSend{batch, event, dest_rank});
}

// ── flush queues where in_flight < limit ───────────────────────────────

void UnifiedDispatcher::_try_flush_queues() {
    for (int r = 0; r < (int)rank_queues_.size(); r++) {
        auto& q = rank_queues_[r];
        if (q.buffered.empty() || q.channel_id < 0) continue;
        if (q.in_flight >= max_in_flight_per_rank_) continue;

        TokenBatch merged = _merge_for_rank(q.buffered);
        q.buffered.clear();
        _do_rank_send(r, merged);
        q.in_flight++;
    }
}

// ── check CUDA events, decrement in_flight ─────────────────────────────

void UnifiedDispatcher::_clean_rank_pending_sends() {
    while (!rank_pending_sends_.empty()) {
        auto& ps = rank_pending_sends_.front();
        cudaError_t err = cudaEventQuery(ps.event);
        if (err == cudaSuccess) {
            CUDACHECK(cudaEventDestroy(ps.event));
            rank_queues_[ps.dest_rank].in_flight--;
            rank_pending_sends_.pop();
        } else if (err == cudaErrorNotReady) {
            break;
        } else {
            DMOE_LOG(ERROR) << "cudaEventQuery failed: " << cudaGetErrorName(err)
                            << ", " << cudaGetErrorString(err) << LEND;
            ASSERT_MSG(false, "Failed to query cuda event");
        }
    }
}

// ── main run loop (overrides MuDispatcher::run) ────────────────────────

void UnifiedDispatcher::run() {
    std::string th_name = "UnifiedDisp@" + std::to_string(this->device_id);
    pthread_setname_np(pthread_self(), th_name.c_str());
#if defined(D_ENABLE_HANG_DEBUGGER) && D_ENABLE_HANG_DEBUGGER == 1
    int timeout4dump = HangDebugger::calcDumpTimeout(this->device_id, 0);
    HangDebugger::registerTimeoutForStackDump(this->thread.native_handle(), timeout4dump, th_name);
#endif
    cudaDeviceSynchronize();

    const auto &make_endpoint = disagmoe::mq_endpoint_factory();
    for (size_t i = 0; i < this->channels.size(); i++) {
        auto endpoint = make_endpoint(this->channels[i]->get_peer_id(), true, -1);
        this->peer_mq[i]->connect(endpoint);
    }

    while (!this->end_flag) {
        _clean_rank_pending_sends();
        _try_flush_queues();

        std::vector<TokenBatch> incoming;
        {
            std::unique_lock<std::mutex> lock(this->mtx);
            if (_has_buffered_sends()) {
                this->cv.wait_for(lock, std::chrono::microseconds(100), [&] {
                    return !this->send_queue.empty() || this->end_flag;
                });
            } else {
                this->cv.wait(lock, [&] {
                    return !this->send_queue.empty() || this->end_flag;
                });
            }
            while (!this->send_queue.empty()) {
                incoming.push_back(this->send_queue.front().first);
                this->send_queue.pop();
            }
        }

        for (auto& batch : incoming) {
            this->_send_once(batch);
        }
        _try_flush_queues();
    }

    // Drain remaining buffered sends on shutdown
    for (auto& q : rank_queues_) {
        if (!q.buffered.empty() && q.channel_id >= 0) {
            TokenBatch merged = _merge_for_rank(q.buffered);
            q.buffered.clear();
            _do_rank_send(q.channel_id, merged);
        }
    }
    // Wait for all pending NCCL sends to complete
    while (!rank_pending_sends_.empty()) {
        auto& ps = rank_pending_sends_.front();
        cudaEventSynchronize(ps.event);
        CUDACHECK(cudaEventDestroy(ps.event));
        rank_pending_sends_.pop();
    }
}

void UnifiedDispatcher::terminate() {
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
