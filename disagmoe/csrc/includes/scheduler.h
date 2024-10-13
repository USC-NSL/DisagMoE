#pragma once

#include <memory>
#include <vector>
#include <string>

#include "comm.h"
#include "muhelper.h"

class Scheduler;

typedef std::shared_ptr<Scheduler> scheduler_t;

class Scheduler {
protected:
    mu_pool_t pool;
    std::vector<int> layer_ids;
    std::string policy;

    std::vector<TensorBatch> _schedule();

public:
    Scheduler(mu_pool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    static scheduler_t build(mu_pool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    TensorBatch schedule();

    void wait_for_new_requests();

    void start();
};


class AttentionScheduler;

typedef std::shared_ptr<AttentionScheduler> attn_scheduler_t;

class AttentionScheduler {
protected:
    mu_attn_pool_t pool;
    std::vector<int> layer_ids;
    std::string policy;

    // virtual std::vector<TensorBatch> _schedule() = 0;

public:
    AttentionScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    static attn_scheduler_t build(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    AttentionBatch schedule();

    void wait_for_new_requests();

    void start();
};