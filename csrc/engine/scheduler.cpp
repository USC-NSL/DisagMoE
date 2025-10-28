#include "scheduler.h"
#include "utils.hpp"
#include "block_manager.h"
#include "cuda_utils.h"
#include "constants.h"
#include "layer.h"

#include <exception>
#include <vector>
#include <string>
#include <set>

// Unified Scheduler implementation

Scheduler::Scheduler(mu_attn_pool_t attn_pool, mu_expert_pool_t expert_pool, std::string policy):
    attn_pool(attn_pool), expert_pool(expert_pool), policy(policy) {
    if (this->attn_pool && !this->expert_pool) {
        // Hardcode policy to MBFLFS for attention
        this->layer_scheduler = std::make_shared<LegacyLayerScheduler>(this->attn_pool->get_num_layers());
        this->attn_pool->set_layer_scheduler(this->layer_scheduler);
    } else if (!this->attn_pool && this->expert_pool) {
        // Hardcode policy to GROUP for experts for now (num_groups=1)
        this->layer_scheduler = std::make_shared<GroupLayerScheduler>(this->expert_pool->get_num_layers(), /*num_groups=*/1);
        this->expert_pool->set_layer_scheduler(this->layer_scheduler);
    } else {
        throw std::runtime_error("Scheduler must be constructed with at least one valid pool");
    }
}

Scheduler::Scheduler(unified_pool_t unified_pool):
    unified_pool(unified_pool) {
    this->layer_scheduler = unified_pool->get_layer_scheduler();
}

void Scheduler::start() {
    if (this->is_attention()) this->attn_pool->start();
    if (this->is_expert()) this->expert_pool->start();
    if (this->is_unified()) this->unified_pool->start();
}

std::vector<int> Scheduler::get_pool_snapshot() {
    // Strict parity with old behavior: only return the cached snapshot
    // captured during the last schedule call.
    return this->pool_snapshot_;
}

// void Scheduler::set_schedule_policy(std::string type) {
//     if (!this->layer_scheduler) {
//         throw std::runtime_error("Layer scheduler is not initialized");
//     }
//     this->layer_scheduler->set_schedule_type(type);
// }

// void Scheduler::set_schedule_block(int step) {
//     if (!this->layer_scheduler) {
//         throw std::runtime_error("Layer scheduler is not initialized");
//     }
//     this->layer_scheduler->set_block_size(step);
// }

void Scheduler::set_schedule_policy(std::string type) {

}

void Scheduler::set_schedule_block(int step) {

}


TokenBatch Scheduler::schedule() {
    if (this->is_attention()) {
        return this->schedule_attention();
    } else if (this->is_expert()) {
        return this->schedule_expert();
    } else if (this->is_unified()) {
        return this->schedule_unified();
    } 
    throw std::runtime_error("Scheduler must be constructed with at least one valid pool");
}

TokenBatch Scheduler::schedule_expert() {
    tx_range _{"Scheduler::schedule_expert"};
    this->pool_snapshot_ = expert_pool->get_pool_snapshot();
    int id = this->layer_scheduler->schedule();
    auto batches = expert_pool->get_batch_from_layer(id);
    auto batch = TokenBatch::merge_by_expert(batches);
    return batch;
}

TokenBatch Scheduler::schedule_attention() {
    tx_range _{"Scheduler::schedule_attention"};
    this->pool_snapshot_ = attn_pool->get_pool_snapshot();
    int id = this->layer_scheduler->schedule();
    auto batches = attn_pool->get_batch_from_layer(id);
    auto batch = TokenBatch::merge_by_attention(batches);
    return batch;
}

TokenBatch Scheduler::schedule_unified() {
    tx_range _{"Scheduler::schedule_unified"};
    this->pool_snapshot_ = {};
    int id = this->layer_scheduler->schedule();
    auto batches = unified_pool->get_batch_from_layer(id);
    auto batch = TokenBatch::merge(batches);
    return batch;
}