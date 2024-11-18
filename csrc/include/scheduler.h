#pragma once

#include <memory>
#include <vector>
#include <string>

#include "comm.h"
#include "muhelper.h"
#include "block_manager.h"
#include "cuda_utils.h"

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

    virtual std::vector<AttentionBatch> _schedule();

public:
    AttentionScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    static attn_scheduler_t build(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy = "largest");

    virtual AttentionBatch schedule();

    // std::vector<int> prepare_block_table_by_meta(attn_metadata_t meta, block_manager_t block_manager);
    // std::vector<int> prepare_block_table(AttentionBatch batch, block_manager_t block_manager);

    void wait_for_new_requests();

    void start();

    virtual std::shared_ptr<NcclGroupChannel> get_channel() {
        return nullptr;
    };
};

class AttentionDriverScheduler : public AttentionScheduler {
protected:
    std::shared_ptr<NcclGroupChannel> chan;
    cudaStream_t stream;

public:
    AttentionDriverScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, Channel_t chan, std::string policy = "largest");

    AttentionBatch schedule() override;

    std::shared_ptr<NcclGroupChannel> get_channel() override;
};

class AttentionWorkerScheduler : public AttentionScheduler {
protected:
    std::shared_ptr<NcclGroupChannel> chan;
    cudaStream_t stream;

public:
    AttentionWorkerScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, Channel_t chan, std::string policy = "largest");

    AttentionBatch schedule() override;

    std::shared_ptr<NcclGroupChannel> get_channel() override;
};