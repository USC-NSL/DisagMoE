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
    std::vector<int> layer_ids;
    std::string policy;
    float cur_queueing_delay{0};
    int max_batch_size{0};
    std::vector<int> pool_snapshot_{};
    std::shared_ptr<LayerSchedulerBase> layer_scheduler;

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
