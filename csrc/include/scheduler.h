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

class Scheduler;

typedef std::shared_ptr<Scheduler> scheduler_t;

class SchedulerBase {
protected:
    mu_pool_t pool;

    std::vector<int> layer_ids;

    std::string policy;

    float cur_queueing_delay;

    int max_batch_size;

    std::vector<int> pool_snapshot_{};

public:
    SchedulerBase(mu_pool_t pool, std::vector<int> layer_ids, std::string policy = "mbfs");

    void start() {
        this->pool->start();
    }

    void wait_for_new_requests() {
        this->pool->wait_for_new_requests();
    }

    void set_max_batch_size(int max_batch_size) {
        this->max_batch_size = max_batch_size;
        this->pool->set_max_batch_size(max_batch_size);
    }

    std::vector<int> get_pool_snapshot() {
        return pool_snapshot_;
    };

    float get_cur_queueing_delay() const {
        return cur_queueing_delay;
    }

    // void set_schedule_policy(std::string policy);

    // void set_schedule_block(int step);
};

class Scheduler: public SchedulerBase {
protected:
    std::vector<TensorBatch> _schedule();

public:
    Scheduler(mu_pool_t pool, std::vector<int> layer_ids, std::string policy = "mbfs");

    static scheduler_t build(mu_pool_t pool, std::vector<int> layer_ids, std::string policy = "mbfs");

    TensorBatch schedule();
};


class AttentionScheduler;

typedef std::shared_ptr<AttentionScheduler> attn_scheduler_t;

class AttentionScheduler: public SchedulerBase {
protected:
    mu_attn_pool_t pool;

    virtual std::vector<AttentionBatch> _schedule();

public:
    AttentionScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy = "mbfs");

    static attn_scheduler_t build(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy = "mbfs");

    virtual AttentionBatch schedule();

    virtual std::shared_ptr<NcclGroupChannel> get_channel() {
        return nullptr;
    };
};

class AttentionDriverScheduler : public AttentionScheduler {
protected:
    // chan is used for intra-group communication in scheduler
    // chan_dist is used for TP group's allreduce
    std::shared_ptr<NcclGroupChannel> chan, chan_dist;

public:
    AttentionDriverScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, Channel_t chan, Channel_t chan_dist, std::string policy = "mbfs");

    AttentionBatch schedule() override;

    std::shared_ptr<NcclGroupChannel> get_channel() override;
};

class AttentionWorkerScheduler : public AttentionScheduler {
protected:
    std::shared_ptr<NcclGroupChannel> chan, chan_dist;
    
    bool end_flag;
    std::thread t_async;
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<AttentionBatch> _schedule_result;

    void async_schedule();

public:
    AttentionWorkerScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, Channel_t chan, Channel_t chan_dist, std::string policy = "mbfs");
    ~AttentionWorkerScheduler();

    AttentionBatch schedule() override;

    std::shared_ptr<NcclGroupChannel> get_channel() override;
};


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
    };

    LayerScheduler(int n_layers, LayerScheduleType type = LayerScheduleType::MBFS);

    virtual int schedule();

    void set_schedule_type(std::string type);

    void set_block_step(int step);

    void remove_tokens_from_layer(int layer_id, int num_tokens) {
        ASSERT(layer_id >= 0 && layer_id < n_layers);
        num_tokens_in_layer[layer_id] -= num_tokens;
        if (num_tokens_in_layer[layer_id] < 0) {
            num_tokens_in_layer[layer_id] = 0;
        }
    }

    virtual void add_tokens_to_layer(int layer_id, int num_tokens);

    virtual void tokens_remain_in_layer(int layer_id, int num_tokens, int num_batches);

protected:
    int n_layers;
    std::vector<int> num_tokens_in_layer;
    std::vector<int> num_batches_in_layer;

private:
    LayerScheduleType type;
    int step;

    /*
        max-batch-first-serve
    */
    int _schedule_mbfs();
    
    /*
        first-layer-first-serve
    */
    int _schedule_flfs();

    /*
        max-block-first-layer-first-serve

        1. Group layers into blocks with step size
        2. Find the block with the largest token count
        3. Find the first layer with tokens in the block

        NOTE(hogura|20250317): 
            * when step=1, this is equivalent to MBFS
            * when step=n_layers, this is equivalent to FLFS
    */
    int _schedule_mbflfs();

    int _schedule_batches_tokens();

    void clean_layer_status(int layer_id) {
        num_tokens_in_layer[layer_id] = 0;
        num_batches_in_layer[layer_id] = 0;
    }

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

    void tokens_remain_in_layer(int layer_id, int num_tokens, int num_batches) override;

};