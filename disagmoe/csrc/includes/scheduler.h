#pragma once

#include <memory>
#include <vector>
#include <string>

#include "comm.h"
#include "muhelper.h"

class Scheduler;

typedef std::shared_ptr<Scheduler> Scheduler_t;

class Scheduler {
protected:
    MuPool_t pool;
    std::vector<int> layer_ids;
    std::string policy;

    virtual std::vector<TensorBatch> _schedule() = 0;

public:
    Scheduler(MuPool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    static Scheduler_t build(MuPool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    TensorBatch merge(std::vector<TensorBatch> batches);

    TensorBatch schedule();

    void wait_for_new_requests();
};

class LargestScheduler: Scheduler {
protected:
    std::vector<TensorBatch> _schedule() override;

public:
    LargestScheduler(MuPool_t pool, std::vector<int> layer_ids);
};