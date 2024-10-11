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
    MuPool_t pool;
    std::vector<int> layer_ids;
    std::string policy;

    virtual std::vector<TensorBatch> _schedule() = 0;

public:
    Scheduler(MuPool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    static scheduler_t build(MuPool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    TensorBatch merge(std::vector<TensorBatch> batches);

    TensorBatch schedule();

    void wait_for_new_requests();

    void start();
};

class LargestScheduler: public Scheduler {
protected:
    std::vector<TensorBatch> _schedule() override;

public:
    LargestScheduler(MuPool_t pool, std::vector<int> layer_ids);
};


typedef std::shared_ptr<AttentionScheduler> attn_scheduler_t;

class AttentionScheduler: {

protected:
    std::vector<BatchTensor> _schedule() override;

public:

    AttentionScheduler(MuPool_t pool, std::vector<int> layer_ids);

    BatchTensor merge(const std::vector<BatchTensor>& batches);

};