#pragma once

#ifndef LAYER_H_
#define LAYER_H_

#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
#include <memory>
#include <vector>
#include <deque>
#include <torch/torch.h>

enum class LayerType { ATTENTION, EXPERT };

class UnifiedLayer;

using unified_layer_t = std::shared_ptr<UnifiedLayer>;

class UnifiedLayer {

private:
    LayerType layer_type;
    int layer_id;
    int expert_id; // >= 0 if is an individual expert
    int num_tokens;

    std::deque<TokenBatch> batch_queue;

    std::deque<TokenTopKInfo> token_queue;

public:
    UnifiedLayer(LayerType layer_type, int layer_id);

    UnifiedLayer(LayerType layer_type, int layer_id, int expert_id);

    static unified_layer_t create_attention_layer(int layer_id);

    static unified_layer_t create_expert_layer(int layer_id);

    inline bool is_attention() const { return layer_type == LayerType::ATTENTION; }

    inline bool is_expert() const { return layer_type == LayerType::EXPERT; }

    inline int get_layer_id() const { return layer_id; }

    inline int get_num_tokens() const { return num_tokens; }

    void add_batch(const TokenBatch &batch);

    void add_batch(torch::Tensor tensor, const batch_metadata_t &meta);

    std::vector<TokenBatch> get_all_batches();

    std::vector<TokenBatch> get_batches_restricted(int token_threshold);

    void add_token(const TokenTopKInfo &token);

    void add_tokens(const std::vector<TokenTopKInfo> &tokens);

    std::vector<TokenTopKInfo> get_all_tokens();

    std::vector<TokenTopKInfo> get_tokens_restricted(int token_threshold);

};


class LayerSchedulerBase {

public:

    virtual int schedule() = 0;

    virtual void add_tokens_to_layer(int layer_id, int num_tokens) = 0;
    
};

/*
 * Base class for unified (attention + MoE collocated) layer schedulers used by UnifiedPool.
 */
class UnifiedLayerSchedulerBase : public LayerSchedulerBase {

protected:

    int num_attn_layers;
    int num_expert_layers;
    int num_layers;  // num_attn_layers + num_expert_layers

    bool attn_use_token_queue;

    int top_k;

    std::shared_ptr<std::mutex> scheduler_mutex;

    std::vector<unified_layer_t> layers; // attn layers first, then expert layers
    std::vector<unified_layer_t> attn_layers;
    std::vector<unified_layer_t> expert_layers;

    virtual bool layer_uses_token_queue(int layer_id);

public:

    // Bundles a scheduling decision with the pre-dequeue pool snapshot,
    // both captured under a single lock acquisition.
    struct ScheduleResult {
        int best_layer;                    // unified layer index (-1 if idle)
        std::vector<int> pool_snapshot;    // queue depths at decision time
    };

    UnifiedLayerSchedulerBase(int num_attn_layers, int num_expert_layers, int topk);

    virtual ~UnifiedLayerSchedulerBase() = default;

    // Query whether a given global layer id corresponds to an attention layer
    // or an expert layer. Semantics are defined by unified schedulers.

    virtual bool is_attn_layer(int layer_id);

    virtual bool is_expert_layer(int layer_id);

    virtual void attn_add_tokens(int layer_id, const std::vector<TokenTopKInfo> &tokens);

    virtual void add_batch(const TokenBatch &batch);

    virtual void add_batch(const torch::Tensor& tensor, const batch_metadata_t &meta);

    virtual std::vector<int> get_pool_snapshot();

    // Schedule and return both the chosen layer and the pool snapshot,
    // captured atomically under a single lock.  Only used when callers
    // need an aligned (decision, snapshot) pair (e.g. advanced logging).
    virtual ScheduleResult schedule_with_snapshot() = 0;

    virtual TokenBatch get_batch_from_layer(int layer_id);

    // Get a batch from `layer_id` but cap the number of tokens to `token_threshold`
    // (<= 0 means no restriction). Implemented by unified schedulers that support
    // partial draining.
    virtual TokenBatch get_batch_from_layer_restricted(int layer_id, int token_threshold);

    void add_tokens_to_layer(int layer_id, int num_tokens) override;
};

// TODO: support expert-wise scheduling
class UnifiedLayerScheduler: public UnifiedLayerSchedulerBase {

public:

    UnifiedLayerScheduler(int num_attn_layers, int num_expert_layers, int topk);

    int schedule() override;

    ScheduleResult schedule_with_snapshot() override;

};

using unified_layer_scheduler_t = std::shared_ptr<UnifiedLayerSchedulerBase>;


class UnifiedDefraggingLayerScheduler : public UnifiedLayerSchedulerBase {

private:

    int lookback_steps;
    int lookahead_steps;
    float weight_decay;


    std::vector<std::queue<int>> history_tokens_in_layer;
    std::vector<int> sum_history_tokens_in_layer;

    // Advance history windows with the latest effective token snapshot.
    void step_end(const std::vector<int> &effective_tokens_snapshot);

    // Helper to convert raw token counts into "fair" effective tokens
    // (MoE layers scaled by 1 / top_k). Attn kept as is.
    inline float get_effective_tokens(int layer_idx, int raw_tokens) const {
        if (raw_tokens <= 0) return 0.0f;
        bool is_attn = (layer_idx < num_attn_layers);
        if (is_attn || top_k <= 1) {
            return static_cast<float>(raw_tokens);
        }
        return static_cast<float>(raw_tokens) / static_cast<float>(top_k);
    }

    int _schedule_impl(std::vector<int>* out_snapshot);

public:
    UnifiedDefraggingLayerScheduler(int num_attn_layers,
                                    int num_expert_layers,
                                    int top_k,
                                    int lookback_steps,
                                    int lookahead_steps,
                                    float weight_decay);

    int schedule() override;

    ScheduleResult schedule_with_snapshot() override;

};


/*

    Layer-wise scheduler

*/



enum LayerSchedulePolicy {
    BASE,
    ADVANCED,
    GROUP,
};

class LegacyLayerScheduler: public LayerSchedulerBase {
public:
    enum LayerScheduleType {
        MBFS,   // max-batch-first-serve
        FLFS,   // first-layer-first-serve
        MBFLFS,  // max-block-first-layer-first-serve
        MBTFS,  // max-batch-token-first-serve
        BIN,   // bin
    };

    LegacyLayerScheduler(int n_layers);

    LegacyLayerScheduler(int n_layers, LayerScheduleType type);

    LegacyLayerScheduler(int n_layers, LayerScheduleType type, int lookback_steps);

    LegacyLayerScheduler(int n_layers, LayerScheduleType type, int lookback_steps, int block_size);

    int schedule() override;

    void set_schedule_type(std::string type);

    void set_block_size(int block_size);

    void remove_tokens_from_layer(int layer_id, int num_tokens) {
        ASSERT(layer_id >= 0 && layer_id < n_layers);
        num_tokens_in_layer[layer_id] -= num_tokens;
        if (num_tokens_in_layer[layer_id] < 0) {
            num_tokens_in_layer[layer_id] = 0;
        }
    }

    void add_tokens_to_layer(int layer_id, int num_tokens) override;

    std::vector<int> get_tokens_per_layer() {
        return num_tokens_in_layer;
    }

    void step_end();


protected:
    int n_layers;
    int lookback_steps;
    constexpr static float weight_decay = 0.8;
    constexpr static int lookahead_steps = 8;
    
    std::vector<int> num_tokens_in_layer;
    std::vector<int> num_batches_in_layer;
    std::vector<std::queue<int>> history_tokens_in_layer;
    std::vector<int> sum_history_tokens_in_layer;

    void clean_layer_status(int layer_id) {
        num_tokens_in_layer[layer_id] = 0;
        num_batches_in_layer[layer_id] = 0;
    }

private:
    LayerScheduleType type;
    int block_size;

    /*
        max-batch-first-serve
    */
    int _schedule_mbfs();
    
    /*
        first-layer-first-serve
    */
    int _schedule_flfs();

    int _schedule_bin();

    /*
        max-block-first-layer-first-serve

        1. Group layers into blocks with block size
        2. Find the block with the largest token count
        3. Find the first layer with tokens in the block

        NOTE(hogura|20250317): 
            * when block_size=1, this is equivalent to MBFS
            * when block_size=n_layers, this is equivalent to FLFS
    */
    int _schedule_mbflfs();

    int _schedule_batches_tokens();

};

class AdvancedLayerScheduler: public LegacyLayerScheduler {

private:
    enum LayerStatus {
        HOLD,
        READY,
        URGENT,
        IDLE,
    };

    int hold_steps;

    std::vector<int> num_steps_to_hold;
    std::vector<long long> ready_timestamp_ms;
    std::vector<LayerStatus> layer_status;

    void set_layer_to_idle(int layer_id) {
        num_tokens_in_layer[layer_id] = 0;
        layer_status[layer_id] = LayerStatus::IDLE;
    }

    void set_layer_to_ready(int layer_id) {
        layer_status[layer_id] = LayerStatus::READY;
        ready_timestamp_ms[layer_id] = t_now_high();
    }

    void set_layer_to_hold(int layer_id) {
        layer_status[layer_id] = LayerStatus::HOLD;
        num_steps_to_hold[layer_id] = hold_steps;
    }

    void set_layer_to_urgent(int layer_id) {
        layer_status[layer_id] = LayerStatus::URGENT;
    }

public:

    AdvancedLayerScheduler(int n_layers, int hold_steps=2);

    int schedule() override; // schedule is protected by a external lock

    void add_tokens_to_layer(int layer_id, int num_tokens) override; // add_tokens_to_layer is protected by a external lock

};

class GroupLayerScheduler: public LegacyLayerScheduler {

private:
    int n_groups;

    inline int get_layer_group_id(int layer_id, int group_id) {
        return layer_id * n_groups + group_id;
    }

    using LegacyLayerScheduler::clean_layer_status;

    void clean_layer_status(int layer_id, int group_id) {
        int layer_group_id = get_layer_group_id(layer_id, group_id);
        clean_layer_status(layer_group_id);
    }

public:

    GroupLayerScheduler(int num_layers, int num_groups);

    GroupLayerScheduler(int num_layers, int num_groups, int lookback_steps);

    int schedule() override;

    using LegacyLayerScheduler::add_tokens_to_layer;

    void add_tokens_to_layer(int layer_id, int group_id, int num_tokens);

};
    

#endif