#pragma once

#include <queue>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <condition_variable>

#include "comm.h"
#include "muhelper.h"
#include "block_manager.h"
#include "cuda_utils.h"
#include "utils.hpp"

/*
    Unified Scheduler holds optional attention and expert pools and a LayerScheduler.
    Single type for both attention and expert scheduling.
*/

class LayerScheduler; // forward declaration

// Note: this Scheduler is not meant to be inherited, and the only
// reason we still have something "virtual" is that we haven't cleanup
// the TP-related classes.
class Scheduler {
protected:
    mu_attn_pool_t attn_pool;
    mu_expert_pool_t expert_pool;
    std::vector<int> layer_ids;
    std::string policy;
    float cur_queueing_delay{0};
    int max_batch_size{0};
    std::vector<int> pool_snapshot_{};
    std::shared_ptr<LayerScheduler> layer_scheduler;

public:
    // shared lock for accessing scheduling states from pools and scheduling logic.
    std::mutex mutex;

    // unified constructor: one or both pools can be null
    Scheduler(mu_attn_pool_t attn_pool, mu_expert_pool_t expert_pool, std::vector<int> layer_ids, std::string policy = "mbfs");

    void start();
    void wait_for_new_requests();

    void set_max_batch_size(int max_batch_size);
    void set_attn_max_batch_size(int max_batch_size);
    void set_expert_max_batch_size(int max_batch_size);
    // General snapshot of current pool state
    std::vector<int> get_pool_snapshot();
    float get_cur_queueing_delay() const { return cur_queueing_delay; }
    void set_schedule_policy(std::string type);
    void set_schedule_block(int step);

    TokenBatch schedule_expert();
    TokenBatch schedule_attention();


    bool has_attention() const { return attn_pool.get() != nullptr; }
    bool has_expert() const { return expert_pool.get() != nullptr; }
};

typedef std::shared_ptr<Scheduler> attn_scheduler_t; // for backward compatibility
typedef std::shared_ptr<Scheduler> scheduler_t;



/*

    Layer-wise scheduler

*/
class LayerScheduler {
public:
    enum LayerScheduleType {
        MBFS,   // max-batch-first-serve
        FLFS,   // first-layer-first-serve
        MBFLFS,  // max-block-first-layer-first-serve
        MBTFS,  // max-batch-token-first-serve
        BIN,   // bin
    };

    LayerScheduler(int n_layers);

    LayerScheduler(int n_layers, LayerScheduleType type);

    LayerScheduler(int n_layers, LayerScheduleType type, int lookback_steps);

    LayerScheduler(int n_layers, LayerScheduleType type, int lookback_steps, int block_size);

    virtual int schedule();

    void set_schedule_type(std::string type);

    void set_block_size(int block_size);

    void remove_tokens_from_layer(int layer_id, int num_tokens) {
        ASSERT(layer_id >= 0 && layer_id < n_layers);
        num_tokens_in_layer[layer_id] -= num_tokens;
        if (num_tokens_in_layer[layer_id] < 0) {
            num_tokens_in_layer[layer_id] = 0;
        }
    }

    virtual void add_tokens_to_layer(int layer_id, int num_tokens);

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

class AdvancedLayerScheduler: public LayerScheduler {

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

class GroupLayerScheduler: public LayerScheduler {

private:
    int n_groups;

    inline int get_layer_group_id(int layer_id, int group_id) {
        return layer_id * n_groups + group_id;
    }

    using LayerScheduler::clean_layer_status;

    void clean_layer_status(int layer_id, int group_id) {
        int layer_group_id = get_layer_group_id(layer_id, group_id);
        clean_layer_status(layer_group_id);
    }

public:

    GroupLayerScheduler(int num_layers, int num_groups);

    GroupLayerScheduler(int num_layers, int num_groups, int lookback_steps);

    int schedule() override;

    using LayerScheduler::add_tokens_to_layer;

    void add_tokens_to_layer(int layer_id, int group_id, int num_tokens);

};
