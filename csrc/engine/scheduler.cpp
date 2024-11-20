#include "scheduler.h"
#include "utils.hpp"
#include "block_manager.h"
#include "cuda_utils.h"

#include <exception>
#include <vector>
#include <string>
#include <set>

scheduler_t Scheduler::build(mu_pool_t pool, std::vector<int> layer_ids, std::string policy) {
    if (policy == "largest") {
        return std::make_shared<Scheduler>(pool, layer_ids);
    } else {
        throw std::runtime_error(policy + " schedule not implemented.");
    }
}


Scheduler::Scheduler(mu_pool_t pool, std::vector<int> layer_ids, std::string policy): 
    pool(pool), layer_ids(layer_ids), policy(policy) {
    
}

void Scheduler::start() {
    this->pool->start();
}

std::vector<TensorBatch> Scheduler::_schedule() {
    return pool->fetch_largest_batch();
}

TensorBatch Scheduler::schedule() {
    tx_range _{"Scheduler::schedule"};

    auto batches = this->_schedule();
    auto batch = TensorBatch::merge(batches);
    return batch;
}

void Scheduler::wait_for_new_requests() {
    pool->wait_for_new_requests();
}



attn_scheduler_t AttentionScheduler::build(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy) {
    if (policy == "largest") {
        return std::make_shared<AttentionScheduler>(pool, layer_ids);
    } else {
        throw std::runtime_error(policy + " schedule not implemented.");
    }
}


AttentionScheduler::AttentionScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy): 
    pool(pool), layer_ids(layer_ids), policy(policy) {
    
}

void AttentionScheduler::start() {
    this->pool->start();
}

std::vector<AttentionBatch> AttentionScheduler::_schedule() {
    return pool->fetch_largest_batch();
}

AttentionBatch AttentionScheduler::schedule() {
    tx_range _{"AttentionScheduler::schedule"};
    
    auto batches = this->_schedule();
    // maybe moving merge to mu_pool results in less memory copy
    auto batch = AttentionBatch::merge(batches);
    return batch;
}

void AttentionScheduler::wait_for_new_requests() {
    pool->wait_for_new_requests();
}

// std::vector<int> AttentionScheduler::prepare_block_table_by_meta(
//     attn_metadata_t meta, block_manager_t block_manager) {
//     return block_manager->prepare_block_table(meta);
// }

// std::vector<int> AttentionScheduler::prepare_block_table(
//     AttentionBatch batch, block_manager_t block_manager) {
//     return prepare_block_table_by_meta(batch.metadata, block_manager);
// }

AttentionDriverScheduler::AttentionDriverScheduler(
    mu_attn_pool_t pool, std::vector<int> layer_ids, 
    Channel_t chan, std::string policy): 
    AttentionScheduler(pool, layer_ids, policy) {
    this->chan = std::dynamic_pointer_cast<NcclGroupChannel>(chan);
    CUDACHECK(cudaStreamCreate(&this->stream));  // unused
}

AttentionBatch AttentionDriverScheduler::schedule() {
    tx_range _{"AttentionDriverScheduler::schedule"};
    int layer_id;
    std::vector<AttentionBatch> batches = pool->fetch_largest_batch(&layer_id);
    if (layer_id == -1) {
        return AttentionBatch{};
    }
    // DMOE_LOG(DEBUG) << "Driver scheduling" << LEND;

    // TODO(hogura|20241119): here only send seq_ids as schedule result; need to send prefill_len

    std::vector<int> schedule_result;
    schedule_result.push_back(layer_id);
    for (auto &batch: batches)
        for (int i: batch.metadata->seq_ids)
            schedule_result.push_back(i);

    DMOE_LOG(DEBUG) << "Driver schedule result: " << layer_id << "; ";
    for (int i = 1; i < schedule_result.size(); i++)
        std::cerr << schedule_result[i] << " ";
    std::cerr << LEND;

    auto cerealized = cerealize_(schedule_result);
    void* buf = cerealized.data();
    size_t size = cerealized.size();
    chan->bcast_obj(buf, size);

    auto batch = AttentionBatch::merge(batches);
    return batch;
}

std::shared_ptr<NcclGroupChannel> AttentionDriverScheduler::get_channel() {
    return chan;
}

AttentionWorkerScheduler::AttentionWorkerScheduler(
    mu_attn_pool_t pool, std::vector<int> layer_ids, 
    Channel_t chan, std::string policy): 
    AttentionScheduler(pool, layer_ids, policy) {
    this->chan = std::dynamic_pointer_cast<NcclGroupChannel>(chan);
    CUDACHECK(cudaStreamCreate(&this->stream));  // unused
}

AttentionBatch AttentionWorkerScheduler::schedule() {
    tx_range _{"AttentionWorkerScheduler::schedule"};
    // DMOE_LOG(DEBUG) << "Worker scheduling" << LEND;
    std::vector<int> schedule_result;
    void* buf;
    size_t size;
    chan->bcast_obj(buf, size);
    decerealize_((char*) buf, size, schedule_result);

    int layer_id = schedule_result[0];
    std::set<int> seq_ids;
    for (int i = 1; i < schedule_result.size(); i++)
        seq_ids.insert(schedule_result[i]);

    DMOE_LOG(DEBUG) << "Worker got result: " << " " << layer_id << "; ";
    for (int i = 1; i < schedule_result.size(); i++)
        std::cerr << schedule_result[i] << " ";
    std::cerr << LEND;

    std::vector<AttentionBatch> batches = pool->fetch_batch_from(layer_id, seq_ids);

    auto batch = AttentionBatch::merge(batches);
    DMOE_LOG(WARNING) << "Worker got batch size: " << batch.metadata->seq_ids.size() << LEND;
    return batch;
}

std::shared_ptr<NcclGroupChannel> AttentionWorkerScheduler::get_channel() {
    return chan;
}