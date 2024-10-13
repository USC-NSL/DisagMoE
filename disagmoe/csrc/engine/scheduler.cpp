#include "scheduler.h"
#include "utils.hpp"

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


// LargestScheduler::LargestScheduler(mu_pool_t pool, std::vector<int> layer_ids): {
//     Scheduler(pool, layer_ids, "largest");
// }

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


// LargestScheduler::LargestScheduler(mu_pool_t pool, std::vector<int> layer_ids): {
//     Scheduler(pool, layer_ids, "largest");
// }

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
    auto batches = this->_schedule();
    // maybe moving merge to mu_pool results in less memory copy
    auto batch = AttentionBatch::merge(batches);
    return batch;
}

void AttentionScheduler::wait_for_new_requests() {
    pool->wait_for_new_requests();
}


