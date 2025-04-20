#include "scheduler.h"
#include "utils.hpp"
#include "block_manager.h"
#include "cuda_utils.h"
#include "constants.h"

#include <exception>
#include <vector>
#include <string>
#include <set>

SchedulerBase::SchedulerBase(mu_pool_t pool, std::vector<int> layer_ids, std::string policy): 
    pool(pool), layer_ids(layer_ids), policy(policy), max_batch_size(MAX_BATCH_SIZE), cur_queueing_delay(0) {
    
}

// void SchedulerBase::set_schedule_policy(std::string policy) {
//     this->policy = policy;
//     this->pool->set_layer_schedule_type(policy);
// }

// void SchedulerBase::set_schedule_block(int step) {
//     this->pool->set_scheduler_block(step);
// }

scheduler_t Scheduler::build(mu_pool_t pool, std::vector<int> layer_ids, std::string policy) {
    return std::make_shared<Scheduler>(pool, layer_ids, policy);
}

Scheduler::Scheduler(mu_pool_t pool, std::vector<int> layer_ids, std::string policy): 
    SchedulerBase(pool, layer_ids, policy) {

}

std::vector<TensorBatch> Scheduler::_schedule() {
    this->pool_snapshot_ = pool->get_pool_snapshot();
    return pool->fetch_largest_batch();
}

TensorBatch Scheduler::schedule() {
    tx_range _{"Scheduler::schedule"};

    auto batches = std::move(this->_schedule());
    auto batch = TensorBatch::merge(batches);
    // if (batch.metadata) {
    //     this->cur_queueing_delay = this->pool->remove_queueing_timer(batch.metadata->req_ids);
    // } else {
    //     this->cur_queueing_delay = 0;
    // }
    return batch;
}

attn_scheduler_t AttentionScheduler::build(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy) {
    if (policy == "mbfs") {
        return std::make_shared<AttentionScheduler>(pool, layer_ids);
    } else {
        throw std::runtime_error(policy + " schedule not implemented.");
    }
}


AttentionScheduler::AttentionScheduler(mu_attn_pool_t pool, std::vector<int> layer_ids, std::string policy): 
    SchedulerBase(pool, layer_ids, policy), pool(pool) {
    
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
    // if (batch.metadata) {
    //     this->cur_queueing_delay = this->pool->remove_queueing_timer(batch.metadata->seq_ids);
    // } else {
    //     this->cur_queueing_delay = 0;
    // }
    return batch;
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

LayerScheduler::LayerScheduler(int n_layers): 
    LayerScheduler(n_layers, LayerScheduleType::FLFS) { }

LayerScheduler::LayerScheduler(int n_layers, LayerScheduler::LayerScheduleType type): 
    LayerScheduler(n_layers, type, 0) { }

LayerScheduler::LayerScheduler(int n_layers, LayerScheduler::LayerScheduleType type, int lookback_steps):
    LayerScheduler(n_layers, type, lookback_steps, 1) { }

LayerScheduler::LayerScheduler(int n_layers, LayerScheduler::LayerScheduleType type, int lookback_steps, int block_size): 
    n_layers(n_layers), type(type), lookback_steps(lookback_steps), block_size(block_size),
    num_tokens_in_layer(std::vector<int>(n_layers, 0)), num_batches_in_layer(std::vector<int>(n_layers, 0)) { 
    if (lookback_steps > 0) {
        history_tokens_in_layer = std::vector<std::queue<int>>(n_layers, std::queue<int>());
        sum_history_tokens_in_layer = std::vector<int>(n_layers, 0);
    }
}

void LayerScheduler::add_tokens_to_layer(int layer_id, int num_tokens) {
    this->num_tokens_in_layer[layer_id] += num_tokens;
    this->num_batches_in_layer[layer_id] += 1;
}

void LayerScheduler::step_end() {
    if (lookback_steps == 0) 
        return;
    for (int i = 0; i < n_layers; i++) {
        if (history_tokens_in_layer[i].size() >= lookback_steps) {
            sum_history_tokens_in_layer[i] -= history_tokens_in_layer[i].front();
            history_tokens_in_layer[i].pop();
        }
        history_tokens_in_layer[i].push(num_tokens_in_layer[i]);
        sum_history_tokens_in_layer[i] += num_tokens_in_layer[i];
    }
}

int LayerScheduler::schedule() {
    int layer_id = -1;
    switch (this->type) {
        case LayerScheduleType::MBFS:
            layer_id = this->_schedule_mbfs();
        case LayerScheduleType::FLFS:
            layer_id = this->_schedule_flfs();
        case LayerScheduleType::MBFLFS:
            layer_id = this->_schedule_mbflfs();
        case LayerScheduleType::MBTFS:
            layer_id = this->_schedule_batches_tokens();
        default:
            throw std::runtime_error("Unknown schedule type.");
    }
    step_end();
    clean_layer_status(layer_id);
    return layer_id;
}

void LayerScheduler::set_schedule_type(std::string type) {
    if (type == "mbfs") {
        this->type = LayerScheduler::LayerScheduleType::MBFS;
    } else if (type == "bin") {
        this->type = LayerScheduler::LayerScheduleType::BIN;
    } else if (type == "flfs") {
        this->type = LayerScheduler::LayerScheduleType::FLFS;
    } else if (type == "mbflfs") {
        this->type = LayerScheduler::LayerScheduleType::MBFLFS;
    } else if (type == "mbtfs") {
        this->type = LayerScheduler::LayerScheduleType::MBTFS;
    } else {
        throw std::runtime_error(type + " schedule not implemented.");
    }
}

void LayerScheduler::set_block_size(int block_size) {
    this->block_size = block_size;
}

int LayerScheduler::_schedule_bin() {
    constexpr int num_threshold = 32;
    int layer_id = -1;
    for (int i = 0; i < n_layers; i++) {
        if (num_tokens_in_layer[i] > 0) {
            layer_id = i;
            break;
        }
    } 
    return layer_id;
}

int LayerScheduler::_schedule_mbfs() {
    int scheduled_layer_id = 0;
    for (int i = 0; i < n_layers; i++) {
        if (num_tokens_in_layer[i] > num_tokens_in_layer[scheduled_layer_id]) {
            scheduled_layer_id = i;
        }
    }
    return scheduled_layer_id;
}

int LayerScheduler::_schedule_flfs() {
    constexpr int num_threshold = 32;
    int layer_id = -1;
    for (int i = 0; i < n_layers; i++) {
        if (num_tokens_in_layer[i] > 0) {
            layer_id = i;
            break;
        }
    } 
    return layer_id;
}

int LayerScheduler::_schedule_mbflfs() {
    // step 1. find the largest block
    int block_i = -1;
    int block_sum = 0;
    for (int i = 0; i < n_layers; i += block_size) {
        int cur_sum = 0;
        for (int j = i; j < std::min(i + block_size, n_layers); j ++)
            cur_sum += num_tokens_in_layer[j];
        if (cur_sum > block_sum) {
            block_sum = cur_sum;
            block_i = i;
        }
    }

    int layer_id = -1;
    // step 2. find the first layer in this block
    for (int i = block_i; i < std::min(block_i + block_size, n_layers); i++) {
        if (num_tokens_in_layer[i] > 0) {
            layer_id = i;
            break;
        }
    }
    return layer_id;
}

int LayerScheduler::_schedule_batches_tokens() {
    int lid = -1;
    int max_batches = 0, max_tokens = 0;

    for (int i = 0; i < n_layers; i ++) {
        int num_batches = num_batches_in_layer[i];
        int num_tokens = num_tokens_in_layer[i];
        if (num_batches > max_batches || (num_batches == max_batches && num_tokens > max_tokens)) {
            lid = i;
            max_batches = num_batches;
            max_tokens = num_tokens;
        }
    }
    return lid;
}

AdvancedLayerScheduler::AdvancedLayerScheduler(int n_layers, int hold_steps):
    LayerScheduler(n_layers),
    hold_steps(hold_steps), layer_status(std::vector<LayerStatus>(n_layers, LayerStatus::IDLE)),
    num_steps_to_hold(std::vector<int>(n_layers, 0)), ready_timestamp_ms(std::vector<long long>(n_layers, 0)) {
}

int AdvancedLayerScheduler::schedule() {
    static float weight_decay = 0.8;
    static int max_wait_time_ms = 100;
    static int lookahead_steps = 8;
    std::vector<int> ready_layers{};
    std::vector<int> urgent_layers{};
    std::vector<int> hold_layers{};

    long long cur_time_ms = t_now_high();

    for (int i = 0; i < n_layers; i++) {
        if (layer_status[i] == LayerStatus::READY) {
            int elapse = static_cast<int>(cur_time_ms - ready_timestamp_ms[i]);
            if (elapse > max_wait_time_ms) {
                // label as urgent
                set_layer_to_urgent(i);
                urgent_layers.emplace_back(i);
            } else {
                ready_layers.emplace_back(i);
            }
        } else if (layer_status[i] == LayerStatus::URGENT) {
            urgent_layers.emplace_back(i);
        } else if (layer_status[i] == LayerStatus::HOLD) {
            hold_layers.emplace_back(i);
        }
    }

    int layer_to_schedule = -1;

    if (urgent_layers.size() > 0) {
        layer_to_schedule = urgent_layers[0];
        long long min_timestamp = ready_timestamp_ms[layer_to_schedule];
        for (int i = 1; i < urgent_layers.size(); i++) {
            int layer_id = urgent_layers[i];
            if (ready_timestamp_ms[layer_id] < min_timestamp) {
                min_timestamp = ready_timestamp_ms[layer_id];
                layer_to_schedule = layer_id;
            }
        }
    } else if (ready_layers.size() > 0) {
        std::vector<float> scores(ready_layers.size());
        for (int i = 0; i < ready_layers.size(); i++) {
            int layer_id = ready_layers[i];
            float decay = 1.f;
            float score = .0f;
            for (int j = 0; j < lookahead_steps; j++) {
                int cur_layer = (layer_id + j) % n_layers;
                score += num_tokens_in_layer[cur_layer] * decay;
                decay *= weight_decay;
            }
            scores[i] = score;
        }
        float max_score = .0f;
        for (int i = 0; i < ready_layers.size(); i++) {
            if (scores[i] > max_score) {
                max_score = scores[i];
                layer_to_schedule = ready_layers[i];
            }
        }
    } else if (hold_layers.size() > 0) {
        layer_to_schedule = hold_layers[0];
    }
    if (layer_to_schedule != -1) {
        for (auto &layer: hold_layers) {
            if (layer == layer_to_schedule) {
                continue;
            }
            num_steps_to_hold[layer]--;
            if (num_steps_to_hold[layer] <= 0) {
                set_layer_to_ready(layer);
            }
        }
    }
    step_end();
    set_layer_to_idle(layer_to_schedule);
    return layer_to_schedule;
}

void AdvancedLayerScheduler::add_tokens_to_layer(int layer_id, int num_tokens) {
    static int THRESHHOLD = 256;
    num_tokens_in_layer[layer_id] += num_tokens;
    if (layer_status[layer_id] == LayerStatus::IDLE) {
        if (num_tokens_in_layer[layer_id] >= THRESHHOLD || hold_steps == 0) {
            set_layer_to_ready(layer_id);
        } else {
            set_layer_to_hold(layer_id);
        }
    } else if (layer_status[layer_id] == LayerStatus::HOLD) {
        if (num_tokens_in_layer[layer_id] >= THRESHHOLD) {
            set_layer_to_ready(layer_id);
        }
    }
}

GroupLayerScheduler::GroupLayerScheduler(int num_layers, int num_groups):
    LayerScheduler(num_layers * num_groups), n_groups(num_groups) { 
    this->n_layers = num_layers;
}

void GroupLayerScheduler::add_tokens_to_layer(int layer_id, int group_id, int num_tokens) {
    int layer_group_id = get_layer_group_id(layer_id, group_id);
    add_tokens_to_layer(layer_group_id, num_tokens);
}

int GroupLayerScheduler::schedule() {
    static float weight_decay = 0.8;
    static int lookahead_steps = 4;
    std::vector<float> scores(n_layers * n_groups);
    for (int i = 0; i < n_layers; i++) {
        float lookahead_score = 0;
        float decay = 1.f;
        for (int k = 0; k < lookahead_steps; k++) {
            int cur_layer = (i + k) % n_layers;
            int num_tokens_cur_layer = 0;
            for (int j = 0; j < n_groups; j++) {
                int layer_group_id = get_layer_group_id(cur_layer, j);
                num_tokens_cur_layer += num_tokens_in_layer[layer_group_id];
                if (lookback_steps > 0) {
                    num_tokens_cur_layer += sum_history_tokens_in_layer[layer_group_id];
                }
            }
            lookahead_score += num_tokens_cur_layer * decay / n_groups;
            decay *= weight_decay;
        }

        for (int j = 0; j < n_groups; j++) {
            int layer_group_id = get_layer_group_id(i, j);
            scores[layer_group_id] = lookahead_score + num_tokens_in_layer[layer_group_id];
        }
    }
    auto max_iter = std::max_element(scores.begin(), scores.end());
    int layer_group_id = std::distance(scores.begin(), max_iter);
    step_end();
    clean_layer_status(layer_group_id);
    return layer_group_id;
}
