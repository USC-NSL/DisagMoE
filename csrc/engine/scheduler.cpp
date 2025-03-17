#include "scheduler.h"
#include "utils.hpp"
#include "block_manager.h"
#include "cuda_utils.h"
#include "constants.h"

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
    pool(pool), layer_ids(layer_ids), policy(policy), max_batch_size(MAX_BATCH_SIZE), cur_queueing_delay(0) {
    
}

void Scheduler::start() {
    this->pool->start();
}

std::vector<TensorBatch> Scheduler::_schedule() {
    this->pool_snapshot_ = pool->get_pool_snapshot();
    return pool->fetch_largest_batch();
}

void Scheduler::set_max_batch_size(int max_batch_size) {
    this->max_batch_size = max_batch_size;
    this->pool->set_max_batch_size(max_batch_size);
}

TensorBatch Scheduler::schedule() {
    tx_range _{"Scheduler::schedule"};

    auto batches = std::move(this->_schedule());
    auto batch = TensorBatch::merge(batches);
    if (batch.metadata) {
        this->cur_queueing_delay = this->pool->remove_queueing_timer(batch.metadata->req_ids);
    } else {
        this->cur_queueing_delay = 0;
    }
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
    pool(pool), layer_ids(layer_ids), policy(policy), max_batch_size(MAX_BATCH_SIZE), cur_queueing_delay(0) {
    
}

void AttentionScheduler::set_max_batch_size(int max_batch_size) {
    this->max_batch_size = max_batch_size;
    this->pool->set_max_batch_size(max_batch_size);
}

void AttentionScheduler::start() {
    this->pool->start();
}

std::vector<AttentionBatch> AttentionScheduler::_schedule() {
    this->pool_snapshot_ = pool->get_pool_snapshot();
    return pool->fetch_largest_batch();
}

AttentionBatch AttentionScheduler::schedule() {
    tx_range _{"AttentionScheduler::schedule"};
    
    auto batches = std::move(this->_schedule());
    // maybe moving merge to mu_pool results in less memory copy
    auto batch = AttentionBatch::merge(batches);
    if (batch.metadata) {
        this->cur_queueing_delay = this->pool->remove_queueing_timer(batch.metadata->seq_ids);
    } else {
        this->cur_queueing_delay = 0;
    }
    return batch;
}

void AttentionScheduler::wait_for_new_requests() {
    pool->wait_for_new_requests();
}

AttentionDriverScheduler::AttentionDriverScheduler(
    mu_attn_pool_t pool, std::vector<int> layer_ids, 
    Channel_t chan, Channel_t chan_dist, std::string policy): 
    AttentionScheduler(pool, layer_ids, policy) {
    this->chan = std::dynamic_pointer_cast<NcclGroupChannel>(chan);
    this->chan_dist = std::dynamic_pointer_cast<NcclGroupChannel>(chan_dist);
}

AttentionBatch AttentionDriverScheduler::schedule() {
    tx_range _{"AttentionDriverScheduler::schedule"};
    int layer_id;
    this->pool_snapshot_ = pool->get_pool_snapshot();
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

    // DMOE_LOG(DEBUG) << "Driver schedule result: " << layer_id << "; ";
    // for (int i = 1; i < schedule_result.size(); i++)
    //     std::cerr << schedule_result[i] << " ";
    // std::cerr << LEND;

    auto cerealized = cerealize_(schedule_result);
    void* buf = cerealized.data();
    size_t size = cerealized.size();
    chan->bcast_obj(buf, size);

    auto batch = AttentionBatch::merge(batches);
    return batch;
}

std::shared_ptr<NcclGroupChannel> AttentionDriverScheduler::get_channel() {
    return chan_dist;
}

AttentionWorkerScheduler::AttentionWorkerScheduler(
    mu_attn_pool_t pool, std::vector<int> layer_ids, 
    Channel_t chan, Channel_t chan_dist, std::string policy): 
    AttentionScheduler(pool, layer_ids, policy) {
    this->chan = std::dynamic_pointer_cast<NcclGroupChannel>(chan);
    this->chan_dist = std::dynamic_pointer_cast<NcclGroupChannel>(chan_dist);
    end_flag = 0;
    this->t_async = std::thread(&AttentionWorkerScheduler::async_schedule, this);
}

AttentionWorkerScheduler::~AttentionWorkerScheduler() {
    this->end_flag = 1;
    this->cv.notify_one();
    this->t_async.join();
}

void AttentionWorkerScheduler::async_schedule() {
    while (!end_flag) {
        tx_range _{"AttentionWorkerScheduler::async_schedule"};
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

        // DMOE_LOG(DEBUG) << "Worker got result: " << " " << layer_id << "; ";
        // for (int i = 1; i < schedule_result.size(); i++)
        //     std::cerr << schedule_result[i] << " ";
        // std::cerr << LEND;

        std::vector<AttentionBatch> batches = pool->fetch_batch_from(layer_id, seq_ids);

        auto batch = AttentionBatch::merge(batches);
        // DMOE_LOG(WARNING) << "Worker got batch size: " << batch.metadata->seq_ids.size() << LEND;

        std::lock_guard lock(this->mutex);
        this->_schedule_result.push(batch);
    }
}

AttentionBatch AttentionWorkerScheduler::schedule() {
    tx_range _{"AttentionWorkerScheduler::schedule"};
    std::lock_guard lock(this->mutex);
    if (this->_schedule_result.empty())
        return AttentionBatch {};
    auto result = this->_schedule_result.front();
    this->_schedule_result.pop();
    return result;
}

std::shared_ptr<NcclGroupChannel> AttentionWorkerScheduler::get_channel() {
    return chan_dist;

}

/*

    Layer-wise scheduler

*/

LayerScheduler::LayerScheduler(mu_pool_t pool, std::vector<int> layer_ids): pool(pool), step(1), n_layers(layer_ids.size())
    {}

int LayerScheduler::schedule() {
    if (pool->get_largest_batch_layer_id() == -1)
        return -1;
    switch (this->type) {
        case LayerScheduleType::MBFS:
            return this->_schedule_mbfs();
        case LayerScheduleType::FLFS:
            return this->_schedule_flfs();
        case LayerScheduleType::MBFLFS:
            return this->_schedule_mbflfs();
        default:
            throw std::runtime_error("Unknown schedule type.");
    }
}

void LayerScheduler::set_schedule_type(LayerScheduler::LayerScheduleType type) {
    this->type = type;
}

void LayerScheduler::set_block_step(int step) {
    this->step = step;
}

int LayerScheduler::_schedule_mbfs() {
    return pool->get_largest_batch_layer_id();
}

int LayerScheduler::_schedule_flfs() {
    for (int i = 0; i < n_layers; i++) {
        if (pool->tokens_in_layer(i) > 0)
            return i;
    }
    return -1;
}

int LayerScheduler::_schedule_mbflfs() {
    // step 1. find the largest block
    int block_i = -1;
    int block_sum = 0;
    for (int i = 0; i < n_layers; i += step) {
        int cur_sum = 0;
        for (int j = i; j < i + step && j < n_layers; j ++)
            cur_sum += pool->tokens_in_layer(j);
        if (cur_sum > block_sum) {
            block_sum = cur_sum;
            block_i = i;
        }
    }

    // step 2. find the first layer in this block
    for (int i = block_i; i < block_i + step && i < n_layers; i++) {
        if (pool->tokens_in_layer(i) > 0)
            return i;
    }

    return -1;
}