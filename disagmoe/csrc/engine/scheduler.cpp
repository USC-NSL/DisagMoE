#include "scheduler.h"
#include "utils.hpp"
#include "block_manager.h"
#include "cuda_utils.h"

#include <exception>
#include <vector>
#include <string>

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

std::vector<std::vector<int>> AttentionScheduler::prepare_block_table_by_meta(
    attn_metadata_t meta, block_manager_t block_manager) {
    AUTO_TX_RANGE;
    // It should be ensured that every seq in batch has been alocated cache blocks
    // For simple case, we allocate cache block in this function, which means every sequence is forcely accepted
    std::vector<std::vector<int>> block_table{};
    int n = meta->seq_ids.size(); // decode seqs are already allocated in previous steps
    for (int i = 0; i < n; i++) {
        block_list_t list{};
        int id = meta->seq_ids[i];
        if (block_manager->has_seq_block_list(id)) {
            list = block_manager->get_seq_block_list(id);
        } else {
            // after implementing waitqueue, we should allocate it in wait_queue
            int seq_len = meta->prefill_seq_len[i];
            list = block_manager->allocate(id, seq_len);
        }
        block_table.emplace_back(*list);
    }
    return block_table;
}

std::vector<std::vector<int>> AttentionScheduler::prepare_block_table(
    AttentionBatch batch, block_manager_t block_manager) {
    return prepare_block_table_by_meta(batch.metadata, block_manager);
}
