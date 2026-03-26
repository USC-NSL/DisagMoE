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

std::vector<int> Scheduler::get_topk_pool_snapshot() {
    if (this->is_unified()) {
        return this->unified_pool->get_topk_pool_snapshot();
    }
    return {};
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

void Scheduler::set_schedule_token_threshold(int attn_token_threshold, int expert_token_threshold) {
    if (this->is_unified()) {
        this->unified_pool->set_attn_schedule_token_threshold(attn_token_threshold);
        this->unified_pool->set_expert_schedule_token_threshold(expert_token_threshold);
    } else {
        throw std::runtime_error("Scheduler must be constructed with a unified pool");
    }
}

TokenBatch Scheduler::schedule() {
    if (this->is_unified()) {
        return this->schedule_unified();
    } else if (this->is_attention()) {
        return this->schedule_attention();
    } else if (this->is_expert()) {
        return this->schedule_expert();
    }
    throw std::runtime_error("Scheduler must be constructed with at least one valid pool");
}

Scheduler::ScheduleTrace Scheduler::schedule_trace() {
    if (this->is_attention()) {
        return this->schedule_trace_attention();
    } else if (this->is_expert()) {
        return this->schedule_trace_expert();
    } else if (this->is_unified()) {
        return this->schedule_trace_unified();
    } 
    throw std::runtime_error("Scheduler must be constructed with at least one valid pool");
}

TokenBatch Scheduler::schedule_expert() {
    int id = this->layer_scheduler->schedule();
    return expert_pool->get_batch_from_layer(id);
}

Scheduler::ScheduleTrace Scheduler::schedule_trace_expert() {
    tx_range _{"Scheduler::schedule_expert"};
    auto snapshot = expert_pool->get_pool_snapshot();
    this->pool_snapshot_ = snapshot;
    int id = this->layer_scheduler->schedule();
    auto batch = expert_pool->get_batch_from_layer(id);
    return ScheduleTrace{batch, std::move(snapshot)};
}

TokenBatch Scheduler::schedule_attention() {
    int id = this->layer_scheduler->schedule();
    return attn_pool->get_batch_from_layer(id);
}

Scheduler::ScheduleTrace Scheduler::schedule_trace_attention() {
    tx_range _{"Scheduler::schedule_attention"};
    auto snapshot = attn_pool->get_pool_snapshot();
    this->pool_snapshot_ = snapshot;
    int id = this->layer_scheduler->schedule();
    auto batch = attn_pool->get_batch_from_layer(id);
    return ScheduleTrace{batch, std::move(snapshot)};
}

TokenBatch Scheduler::schedule_unified() {
    int id = this->layer_scheduler->schedule();
    return unified_pool->get_batch_from_layer(id);
}

Scheduler::ScheduleTrace Scheduler::schedule_trace_unified() {
    tx_range _{"Scheduler::schedule_unified"};
    auto result = unified_pool->schedule_with_snapshot();
    this->pool_snapshot_ = result.pool_snapshot;
    auto batch = unified_pool->get_batch_from_layer(result.best_layer);
    return ScheduleTrace{batch, std::move(result.pool_snapshot)};
}
