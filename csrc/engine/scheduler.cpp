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


// void SchedulerBase::set_schedule_policy(std::string policy) {
//     this->policy = policy;
//     this->pool->set_layer_schedule_type(policy);
// }

// void SchedulerBase::set_schedule_block(int step) {
//     this->pool->set_scheduler_block(step);
// }

// Unified Scheduler implementation

Scheduler::Scheduler(mu_attn_pool_t attn_pool, mu_expert_pool_t expert_pool, std::vector<int> layer_ids, std::string policy):
    attn_pool(attn_pool), expert_pool(expert_pool), layer_ids(layer_ids), policy(policy), max_batch_size(MAX_BATCH_SIZE), cur_queueing_delay(0) {
    if (this->attn_pool && !this->expert_pool) {
        // Hardcode policy to MBFLFS for attention
        this->layer_scheduler = std::make_shared<AdvancedLayerScheduler>(this->attn_pool->get_num_layers(), 0);
        this->attn_pool->set_layer_scheduler(this->layer_scheduler);
    } else if (!this->attn_pool && this->expert_pool) {
        // Hardcode policy to GROUP for experts for now (num_groups=1)
        this->layer_scheduler = std::make_shared<GroupLayerScheduler>(this->expert_pool->get_num_layers(), /*num_groups=*/1);
        this->expert_pool->set_layer_scheduler(this->layer_scheduler);
    } else if (this->attn_pool && this->expert_pool) {
        // Future: colocated pools goes to this path
        this->layer_scheduler = std::make_shared<LegacyLayerScheduler>((int)layer_ids.size(), LegacyLayerScheduler::LayerScheduleType::MBFLFS);
        this->attn_pool->set_layer_scheduler(this->layer_scheduler);
        this->expert_pool->set_layer_scheduler(this->layer_scheduler);
    } else {
        throw std::runtime_error("Scheduler must be constructed with at least one valid pool");
    }
}

void Scheduler::start() {
    if (this->attn_pool) this->attn_pool->start();
    if (this->expert_pool) this->expert_pool->start();
}

void Scheduler::wait_for_new_requests() {
    if (this->attn_pool) this->attn_pool->wait_for_new_requests();
    if (this->expert_pool) this->expert_pool->wait_for_new_requests();
}

void Scheduler::set_max_batch_size(int max_batch_size) {
    this->max_batch_size = max_batch_size;
    if (this->attn_pool) this->attn_pool->set_max_batch_size(max_batch_size);
    if (this->expert_pool) this->expert_pool->set_max_batch_size(max_batch_size);
}

void Scheduler::set_attn_max_batch_size(int max_batch_size) {
    if (this->attn_pool) this->attn_pool->set_max_batch_size(max_batch_size);
}

void Scheduler::set_expert_max_batch_size(int max_batch_size) {
    if (this->expert_pool) this->expert_pool->set_max_batch_size(max_batch_size);
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


TokenBatch Scheduler::schedule_expert() {
    tx_range _{"Scheduler::schedule_expert"};
    if (!this->expert_pool) return TokenBatch{};
    std::lock_guard lock(this->mutex);
    this->pool_snapshot_ = expert_pool->get_pool_snapshot();
    int id = this->layer_scheduler->schedule();
    auto batches = expert_pool->get_batch_from_layer(id);
    auto batch = TokenBatch::merge_by_expert(batches);
    return batch;
}

TokenBatch Scheduler::schedule_attention() {
    tx_range _{"Scheduler::schedule_attention"};
    if (!this->attn_pool) return TokenBatch{};
    std::lock_guard lock(this->mutex);
    this->pool_snapshot_ = attn_pool->get_pool_snapshot();
    int id = this->layer_scheduler->schedule();
    auto batches = attn_pool->get_batch_from_layer(id);
    auto batch = TokenBatch::merge_by_attention(batches);
    return batch;
}