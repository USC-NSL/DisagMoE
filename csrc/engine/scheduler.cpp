#include "scheduler.h"
#include "utils.hpp"
#include "block_manager.h"
#include "cuda_utils.h"
#include "constants.h"

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
        this->layer_scheduler = std::make_shared<LayerScheduler>((int)layer_ids.size(), LayerScheduler::LayerScheduleType::MBFLFS);
        this->attn_pool->set_layer_scheduler(this->layer_scheduler);
    } else if (!this->attn_pool && this->expert_pool) {
        // Hardcode policy to GROUP for experts for now (num_groups=1)
        this->layer_scheduler = std::make_shared<GroupLayerScheduler>((int)layer_ids.size(), /*num_groups=*/1);
        this->expert_pool->set_layer_scheduler(this->layer_scheduler);
    } else if (this->attn_pool && this->expert_pool) {
        // Future: colocated pools goes to this path
        this->layer_scheduler = std::make_shared<LayerScheduler>((int)layer_ids.size(), LayerScheduler::LayerScheduleType::MBFLFS);
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

void Scheduler::set_schedule_policy(std::string type) {
    if (!this->layer_scheduler) {
        throw std::runtime_error("Layer scheduler is not initialized");
    }
    this->layer_scheduler->set_schedule_type(type);
}

void Scheduler::set_schedule_block(int step) {
    if (!this->layer_scheduler) {
        throw std::runtime_error("Layer scheduler is not initialized");
    }
    this->layer_scheduler->set_block_size(step);
}

TensorBatch Scheduler::schedule_expert() {
    tx_range _{"Scheduler::schedule_expert"};
    if (!this->expert_pool) return TensorBatch{};
    std::lock_guard lock(this->mutex);
    this->pool_snapshot_ = expert_pool->get_pool_snapshot();
    int id = this->layer_scheduler->schedule();
    auto batches = expert_pool->get_batch_from_layer(id);
    auto batch = TensorBatch::merge(batches);
    return batch;
}

AttentionBatch Scheduler::schedule_attention() {
    tx_range _{"Scheduler::schedule_attention"};
    if (!this->attn_pool) return AttentionBatch{};
    std::lock_guard lock(this->mutex);
    this->pool_snapshot_ = attn_pool->get_pool_snapshot();
    int id = this->layer_scheduler->schedule();
    auto batches = attn_pool->get_batch_from_layer(id);
    auto batch = AttentionBatch::merge(batches);
    return batch;
}

AttentionDriverScheduler::AttentionDriverScheduler(
    mu_attn_pool_t pool, std::vector<int> layer_ids, 
    Channel_t chan, Channel_t chan_dist, std::string policy): 
    Scheduler(pool, nullptr, layer_ids, policy) {
    this->chan = std::dynamic_pointer_cast<NcclGroupChannel>(chan);
    this->chan_dist = std::dynamic_pointer_cast<NcclGroupChannel>(chan_dist);
}

AttentionBatch AttentionDriverScheduler::schedule_attention() {
    tx_range _{"AttentionDriverScheduler::schedule_attention"};
    this->pool_snapshot_ = attn_pool->get_pool_snapshot();
    int layer_id = this->layer_scheduler->schedule();
    std::vector<AttentionBatch> batches = attn_pool->get_batch_from_layer(layer_id);
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

std::shared_ptr<NcclGroupChannel> AttentionDriverScheduler::get_attention_channel() {
    return chan_dist;
}


AttentionWorkerScheduler::AttentionWorkerScheduler(
    mu_attn_pool_t pool, std::vector<int> layer_ids, 
    Channel_t chan, Channel_t chan_dist, std::string policy): 
    Scheduler(pool, nullptr, layer_ids, policy) {
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

        std::vector<AttentionBatch> batches = attn_pool->fetch_batch_from(layer_id, seq_ids);

        auto batch = AttentionBatch::merge(batches);
        // DMOE_LOG(WARNING) << "Worker got batch size: " << batch.metadata->seq_ids.size() << LEND;

        std::lock_guard lock(this->mutex);
        this->_schedule_result.push(batch);
    }
}

AttentionBatch AttentionWorkerScheduler::schedule_attention() {
    tx_range _{"AttentionWorkerScheduler::schedule_attention"};
    std::lock_guard lock(this->mutex);
    if (this->_schedule_result.empty())
        return AttentionBatch {};
    auto result = this->_schedule_result.front();
    this->_schedule_result.pop();
    return result;
}

std::shared_ptr<NcclGroupChannel> AttentionWorkerScheduler::get_attention_channel() {
    return chan_dist;
}


/*

    Layer-wise scheduler

*/

LayerScheduler::LayerScheduler(int n_layers, LayerScheduler::LayerScheduleType type): 
    n_layers(n_layers), block_size(8), type(type), 
    num_tokens_in_layer(std::vector<int>(n_layers, 0)), num_batches_in_layer(std::vector<int>(n_layers, 0)) { }

void LayerScheduler::add_tokens_to_layer(int layer_id, int num_tokens) {
    this->num_tokens_in_layer[layer_id] += num_tokens;
    this->num_batches_in_layer[layer_id] += 1;
}

int LayerScheduler::schedule() {
    switch (this->type) {
        case LayerScheduleType::MBFS:
            return this->_schedule_mbfs();
        case LayerScheduleType::FLFS:
            return this->_schedule_flfs();
        case LayerScheduleType::MBFLFS:
            return this->_schedule_mbflfs();
        case LayerScheduleType::MBTFS:
            return this->_schedule_batches_tokens();
        default:
            throw std::runtime_error("Unknown schedule type.");
    }
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
    clean_layer_status(layer_id);
    return layer_id;
}

int LayerScheduler::_schedule_mbfs() {
    int scheduled_layer_id = 0;
    for (int i = 0; i < n_layers; i++) {
        if (num_tokens_in_layer[i] > num_tokens_in_layer[scheduled_layer_id]) {
            scheduled_layer_id = i;
        }
    }
    clean_layer_status(scheduled_layer_id);
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
    clean_layer_status(layer_id);
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
    clean_layer_status(layer_id);
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
    clean_layer_status(lid);
    return lid;
}

AdvancedLayerScheduler::AdvancedLayerScheduler(int n_layers, int hold_steps):
    LayerScheduler(n_layers),
    hold_steps(hold_steps), layer_status(std::vector<LayerStatus>(n_layers, LayerStatus::IDLE)),
    num_steps_to_hold(std::vector<int>(n_layers, 0)), ready_timestamp_ms(std::vector<long long>(n_layers, 0)) {
}

int AdvancedLayerScheduler::schedule() {
    static float weight_decay = 0.8;
    static int max_wait_time_ms = 30;
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
            float decay = 1;
            float score = .0f;
            for (int j = 0; j < 4; j++) {
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
    std::vector<float> scores(n_layers * n_groups);
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < n_groups; j++) {
            int layer_group_id = get_layer_group_id(i, j);
            float decay = 1;
            float score = .0f;
            for (int k = 0; k < 4; k++) {
                int cur_layer = (i + k) % n_layers;
                int cur_layer_group_id = get_layer_group_id(cur_layer, j);
                score += num_tokens_in_layer[cur_layer_group_id] * decay;
                decay *= weight_decay;
            }
            scores[layer_group_id] = score;
        }
    }
    auto max_iter = std::max_element(scores.begin(), scores.end());
    int layer_group_id = std::distance(scores.begin(), max_iter);
    clean_layer_status(layer_group_id);
    return layer_group_id;
}
