#include "scheduler.h"
#include "utils.hpp"

#include <exception>
#include <vector>

scheduler_t Scheduler::build(MuPool_t pool, std::vector<int> layer_ids, std::string policy) {
    if (policy == "largest") {
        return std::make_shared<LargestScheduler>(pool, layer_ids);
    } else {
        throw std::runtime_error(policy + " schedule not implemented.");
    }
}

LargestScheduler::LargestScheduler(MuPool_t pool, std::vector<int> layer_ids):
    Scheduler(pool, layer_ids, "largest") {

}

Scheduler::Scheduler(MuPool_t pool, std::vector<int> layer_ids, std::string policy): 
    pool(pool), layer_ids(layer_ids), policy(policy) {
    
}

void Scheduler::start() {
    this->pool->start();
}

TensorBatch Scheduler::schedule() {
    auto batches = this->_schedule();
    auto batch = TensorBatch::merge(batches);
    return batch;
}

void Scheduler::wait_for_new_requests() {
    pool->wait_for_new_requests();
}

std::vector<TensorBatch> LargestScheduler::_schedule() {
    return pool->fetch_largest_batch();
}