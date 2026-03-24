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
#include "layer.h"
#include "pool.h"

/*
    Unified Scheduler holds optional attention and expert pools and a LayerScheduler.
    Single type for both attention and expert scheduling.
*/

// Note: this Scheduler is not meant to be inherited, and the only
// reason we still have something "virtual" is that we haven't cleanup
// the TP-related classes.
class Scheduler {

protected:
    mu_attn_pool_t attn_pool;
    mu_expert_pool_t expert_pool;
    unified_pool_t unified_pool;

    std::string policy;
    std::vector<int> pool_snapshot_{};
    std::shared_ptr<LayerSchedulerBase> layer_scheduler;

public:
    // unified constructor: one or both pools can be null
    Scheduler(mu_attn_pool_t attn_pool, mu_expert_pool_t expert_pool, std::string policy = "mbfs");

    Scheduler(unified_pool_t unified_pool);

    void start();

    // General snapshot of current pool state
    std::vector<int> get_pool_snapshot();
    std::vector<int> get_topk_pool_snapshot();
    void set_schedule_policy(std::string type);
    void set_schedule_block(int step);
    void set_schedule_token_threshold(int attn_token_threshold, int expert_token_threshold);

    // Bundles a scheduling result with the pre-schedule pool snapshot taken
    // atomically in the same call, so callers get aligned (batch, snapshot) pairs.
    struct ScheduleTrace {
        TokenBatch batch;                  // the scheduled batch (may be empty)
        std::vector<int> pool_snapshot;    // queue depths captured before dequeue
    };

    // Schedule the next batch; internally delegates to schedule_trace().
    TokenBatch schedule();
    // Schedule and return both the batch and the pre-schedule pool snapshot.
    ScheduleTrace schedule_trace();

    // Pool-specific schedule helpers (return batch only).
    TokenBatch schedule_expert();
    TokenBatch schedule_attention();
    TokenBatch schedule_unified();

    // Pool-specific schedule helpers that also capture the aligned snapshot.
    ScheduleTrace schedule_trace_expert();
    ScheduleTrace schedule_trace_attention();
    ScheduleTrace schedule_trace_unified();
    
    inline bool is_attention() const { return attn_pool.get() != nullptr; }
    inline bool is_expert() const { return expert_pool.get() != nullptr; }
    inline bool is_unified() const { return unified_pool.get() != nullptr; }
};

typedef std::shared_ptr<Scheduler> scheduler_t;
