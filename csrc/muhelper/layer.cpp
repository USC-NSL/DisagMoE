#include "layer.h"
#include "batch.hpp"

UnifiedLayer::UnifiedLayer(LayerType layer_type, int layer_id): 
    layer_type(layer_type), layer_id(layer_id), expert_id(-1), num_tokens(0), num_batches(0) {}

UnifiedLayer::UnifiedLayer(LayerType layer_type, int layer_id, int expert_id): 
    layer_type(layer_type), layer_id(layer_id), expert_id(expert_id), num_tokens(0), num_batches(0) {}

unified_layer_t UnifiedLayer::create_attention_layer(int layer_id) {
    return std::make_shared<UnifiedLayer>(LayerType::ATTENTION, layer_id);
}

unified_layer_t UnifiedLayer::create_expert_layer(int layer_id) {
    return std::make_shared<UnifiedLayer>(LayerType::EXPERT, layer_id);
}

void UnifiedLayer::add_batch(const TokenBatch &batch) {
    this->batch_queue.push_back(batch);
    this->num_tokens += batch.metadata->num_tokens();
    this->num_batches += 1;
}

void UnifiedLayer::add_batch(torch::Tensor data, const batch_metadata_t &meta) {
    this->batch_queue.emplace_back(data, meta);
    this->num_tokens += meta->num_tokens();
    this->num_batches += 1;
}

std::vector<TokenBatch> UnifiedLayer::get_all_batches() {
    std::vector<TokenBatch> result{};
    result.reserve(this->batch_queue.size());
    while (!this->batch_queue.empty()) {
        result.emplace_back(std::move(this->batch_queue.front()));
        this->batch_queue.pop_front();
    }
    this->num_tokens = 0;
    this->num_batches = 0;
    return result;
}

std::vector<TokenBatch> UnifiedLayer::get_batches_restricted(int token_threshold) {
    std::vector<TokenBatch> result{};
    int total_tokens = 0;
    int num_batches = 0;
    while (!this->batch_queue.empty() && total_tokens < token_threshold) {
        auto first_batch = this->batch_queue.front();
        int tokens_in_batch = first_batch.metadata->num_tokens();
        if (total_tokens + tokens_in_batch > token_threshold) {
            int need_tokens = token_threshold - total_tokens;
            auto batches = first_batch.split_with_sizes({need_tokens, tokens_in_batch - need_tokens});
            tokens_in_batch = batches[0].metadata->num_tokens();
            result.emplace_back(std::move(batches[0]));
            this->batch_queue.front() = std::move(batches[1]);
        } else {
            result.emplace_back(std::move(this->batch_queue.front()));
            this->batch_queue.pop_front();
        }
        total_tokens += tokens_in_batch;
        num_batches += 1;
    }
    this->num_tokens -= total_tokens;
    this->num_batches -= num_batches;
    ASSERT_MSG(total_tokens > 0 && num_batches > 0, "Got nothing from layer" + std::to_string(this->layer_id) + \
    " under token threshold, total tokens in layer: " + std::to_string(this->num_tokens) + \
    ", num tokens in next batch in layer: " + std::to_string(this->batch_queue.front().metadata->num_tokens()) + \
    ", token threshold: " + std::to_string(token_threshold));
    return result;
}

/*

    Layer-wise scheduler

*/

UnifiedLayerScheduler::UnifiedLayerScheduler(int num_attn_layers, int num_expert_layers):
    num_attn_layers(num_attn_layers), num_expert_layers(num_expert_layers) {
    for (int i = 0; i < num_attn_layers; i++) {
        auto attn_layer = UnifiedLayer::create_attention_layer(i);
        this->attn_layers.push_back(attn_layer);
        this->layers.push_back(attn_layer);
    }

    for (int i = 0; i < num_expert_layers; i++) {
        auto expert_layer = UnifiedLayer::create_expert_layer(i);
        this->expert_layers.push_back(expert_layer);
        this->layers.push_back(expert_layer);
    }
}

UnifiedLayerScheduler::UnifiedLayerScheduler(int num_layers):
    UnifiedLayerScheduler(num_layers, num_layers) { }

bool UnifiedLayerScheduler::is_attn_layer(int layer_id) {
    return layer_id < this->num_attn_layers;
}

bool UnifiedLayerScheduler::is_expert_layer(int layer_id) {
    return layer_id >= this->num_attn_layers;
}

int UnifiedLayerScheduler::schedule() {
    // TODO: this is fisrt-layer-first-serve, implement other policies
    for (int i = 0; i < this->num_attn_layers; i++) {
        if (this->attn_layers[i]->get_num_tokens() > 0) {
            return this->attn_layers[i]->get_layer_id();
        }
        if (i < this->num_expert_layers && this->expert_layers[i]->get_num_tokens() > 0) {
            return this->expert_layers[i]->get_layer_id() + num_attn_layers;
        }
    }

    return -1;
}

TokenBatch UnifiedLayerScheduler::get_batch_from_layer(int layer_id) {
    if (layer_id < 0 || layer_id >= (this->num_attn_layers + this->num_expert_layers)) {
        return TokenBatch {};
    }
    auto batches = this->layers[layer_id]->get_all_batches();
    auto batch = TokenBatch::merge(batches);
    return batch;
}

TokenBatch UnifiedLayerScheduler::get_batch_from_layer_restricted(int layer_id, int token_threshold) {
    if (layer_id < 0 || layer_id >= (this->num_attn_layers + this->num_expert_layers)) {
        return TokenBatch {};
    }
    if (token_threshold > 0 && this->layers[layer_id]->get_num_tokens() > token_threshold) {
        auto batches = this->layers[layer_id]->get_batches_restricted(token_threshold);
        auto batch = TokenBatch::merge(batches);
        return batch;
    } 
    return this->get_batch_from_layer(layer_id);
}

std::vector<int> UnifiedLayerScheduler::get_pool_snapshot() {
    std::vector<int> snapshot(this->num_attn_layers + this->num_expert_layers, 0);
    for (int i = 0; i < this->num_attn_layers; i++) {
        snapshot[i] = this->attn_layers[i]->get_num_tokens();
    }
    for (int i = 0; i < this->num_expert_layers; i++) {
        snapshot[i + this->num_attn_layers] = this->expert_layers[i]->get_num_tokens();
    }
    return snapshot;
}

void UnifiedLayerScheduler::add_tokens_to_layer(int layer_id, int num_tokens) {
    throw std::runtime_error("add_tokens_to_layer is not supported for UnifiedLayerScheduler");
}

void UnifiedLayerScheduler::add_batch(const TokenBatch &batch) {
    if (batch.metadata->is_attention()) {
        this->attn_layers[batch.metadata->layer_id]->add_batch(batch);
    } else if (batch.metadata->is_expert()) {
        this->expert_layers[batch.metadata->layer_id]->add_batch(batch);
    } else {
        ASSERT_MSG(false, "Invalid batch metadata");
    }
}

void UnifiedLayerScheduler::add_batch(const torch::Tensor& tensor, const batch_metadata_t &meta) {
    if (meta->is_attention()) {
        this->attn_layers[meta->layer_id]->add_batch(tensor, meta);
    } else if (meta->is_expert()) {
        this->expert_layers[meta->layer_id]->add_batch(tensor, meta);
    } else {
        ASSERT_MSG(false, "Invalid batch metadata");
    }
}

/*

    Unified defragging layer scheduler

*/

UnifiedDefraggingLayerScheduler::UnifiedDefraggingLayerScheduler(
    int num_attn_layers, int num_expert_layers, int top_k,
    int lookback_steps, int lookahead_steps, float weight_decay):
    num_attn_layers(num_attn_layers),
    num_expert_layers(num_expert_layers),
    num_layers(num_attn_layers + num_expert_layers),
    top_k(top_k > 0 ? top_k : 1),
    lookback_steps(lookback_steps),
    lookahead_steps(lookahead_steps),
    weight_decay(weight_decay) {

    this->layers.reserve(num_layers);

    for (int i = 0; i < num_attn_layers; i++) {
        auto attn_layer = UnifiedLayer::create_attention_layer(i);
        this->attn_layers.push_back(attn_layer);
        this->layers.push_back(attn_layer);
    }

    for (int i = 0; i < num_expert_layers; i++) {
        auto expert_layer = UnifiedLayer::create_expert_layer(i);
        this->expert_layers.push_back(expert_layer);
        this->layers.push_back(expert_layer);
    }

    if (this->lookback_steps > 0) {
        this->history_tokens_in_layer = std::vector<std::queue<int>>(this->num_layers, std::queue<int>());
        this->sum_history_tokens_in_layer = std::vector<int>(this->num_layers, 0);
    }
}

UnifiedDefraggingLayerScheduler::UnifiedDefraggingLayerScheduler(
    int num_layers, int top_k, int lookback_steps,
    int lookahead_steps, float weight_decay):
    UnifiedDefraggingLayerScheduler(
        num_layers, num_layers, top_k, lookback_steps, lookahead_steps, weight_decay) { }

bool UnifiedDefraggingLayerScheduler::is_attn_layer(int layer_id) {
    return layer_id < this->num_attn_layers;
}

bool UnifiedDefraggingLayerScheduler::is_expert_layer(int layer_id) {
    return layer_id >= this->num_attn_layers;
}

void UnifiedDefraggingLayerScheduler::step_end(const std::vector<int> &effective_tokens_snapshot) {
    if (lookback_steps == 0) {
        return;
    }
    if (history_tokens_in_layer.empty()) {
        history_tokens_in_layer = std::vector<std::queue<int>>(num_layers, std::queue<int>());
        sum_history_tokens_in_layer = std::vector<int>(num_layers, 0);
    }

    for (int i = 0; i < num_layers; i++) {
        auto &hist_q = history_tokens_in_layer[i];
        if (static_cast<int>(hist_q.size()) >= lookback_steps) {
            sum_history_tokens_in_layer[i] -= hist_q.front();
            hist_q.pop();
        }
        hist_q.push(effective_tokens_snapshot[i]);
        sum_history_tokens_in_layer[i] += effective_tokens_snapshot[i];
    }
}

int UnifiedDefraggingLayerScheduler::schedule() {
    std::vector<int> raw_tokens(num_layers, 0);
    std::vector<float> effective_tokens(num_layers, 0.0f);
    for (int i = 0; i < num_layers; i++) {
        int tokens = layers[i]->get_num_tokens();
        raw_tokens[i] = tokens;
        effective_tokens[i] = get_effective_tokens(i, tokens);
    }

    // Early exit if there is no work to schedule.
    int total_raw = 0;
    for (int t : raw_tokens) total_raw += t;
    if (total_raw == 0) {
        return -1;
    }

    std::vector<float> scores(num_layers, 0.0f);

    // Compute scores for attention layers (indices 0 to num_attn_layers - 1)
    for (int i = 0; i < num_attn_layers; i++) {
        float lookahead_score = 0.0f;
        float decay = weight_decay;

        for (int k = 1; k < lookahead_steps && k < num_attn_layers; k++) {
            int cur_layer = (i + k) % num_attn_layers;
            float num_tokens_cur_layer = effective_tokens[cur_layer];
            float history_score = 0.0f;

            if (lookback_steps > 0 &&
                !history_tokens_in_layer.empty() &&
                !history_tokens_in_layer[cur_layer].empty() &&
                sum_history_tokens_in_layer[cur_layer] > 0) {
                int window_size =
                    static_cast<int>(history_tokens_in_layer[cur_layer].size());
                if (window_size > 0) {
                    history_score =
                        static_cast<float>(sum_history_tokens_in_layer[cur_layer]) /
                        static_cast<float>(window_size);
                }
            }

            lookahead_score += (num_tokens_cur_layer + history_score) * decay;
            decay *= weight_decay;
        }

        float immediate = effective_tokens[i];
        if (immediate > 0.0f) {
            scores[i] = lookahead_score + immediate;
        } else {
            scores[i] = 0.0f;
        }
    }

    // Compute scores for expert layers (indices num_attn_layers to num_layers - 1)
    for (int i = num_attn_layers; i < num_layers; i++) {
        float lookahead_score = 0.0f;
        float decay = weight_decay;

        for (int k = 1; k < lookahead_steps && k < num_expert_layers; k++) {
            // Wrap within expert layer range
            int offset_in_expert = i - num_attn_layers;
            int cur_offset = (offset_in_expert + k) % num_expert_layers;
            int cur_layer = num_attn_layers + cur_offset;
            float num_tokens_cur_layer = effective_tokens[cur_layer];
            float history_score = 0.0f;

            if (lookback_steps > 0 &&
                !history_tokens_in_layer.empty() &&
                !history_tokens_in_layer[cur_layer].empty() &&
                sum_history_tokens_in_layer[cur_layer] > 0) {
                int window_size =
                    static_cast<int>(history_tokens_in_layer[cur_layer].size());
                if (window_size > 0) {
                    history_score =
                        static_cast<float>(sum_history_tokens_in_layer[cur_layer]) /
                        static_cast<float>(window_size);
                }
            }

            lookahead_score += (num_tokens_cur_layer + history_score) * decay;
            decay *= weight_decay;
        }

        float immediate = effective_tokens[i];
        if (immediate > 0.0f) {
            scores[i] = lookahead_score + immediate;
        } else {
            scores[i] = 0.0f;
        }
    }

    // Choose the layer with the largest score.
    float max_score = 0.0f;
    int best_layer = -1;
    for (int i = 0; i < num_layers; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
            best_layer = i;
        }
    }

    // Update history windows with the effective token snapshot.
    std::vector<int> effective_snapshot_int(num_layers, 0);
    for (int i = 0; i < num_layers; i++) {
        // We store integer approximations for history, consistent with
        // LegacyLayerScheduler.
        effective_snapshot_int[i] = static_cast<int>(effective_tokens[i]);
    }
    step_end(effective_snapshot_int);

    return best_layer;
}

void UnifiedDefraggingLayerScheduler::add_tokens_to_layer(int layer_id, int num_tokens) {
    throw std::runtime_error("add_tokens_to_layer is not supported for UnifiedDefraggingLayerScheduler");
}

void UnifiedDefraggingLayerScheduler::add_batch(const TokenBatch &batch) {
    if (batch.metadata->is_attention()) {
        this->attn_layers[batch.metadata->layer_id]->add_batch(batch);
    } else if (batch.metadata->is_expert()) {
        this->expert_layers[batch.metadata->layer_id]->add_batch(batch);
    } else {
        ASSERT_MSG(false, "Invalid batch metadata");
    }
}

void UnifiedDefraggingLayerScheduler::add_batch(
    const torch::Tensor &tensor,
    const batch_metadata_t &meta
) {
    if (meta->is_attention()) {
        this->attn_layers[meta->layer_id]->add_batch(tensor, meta);
    } else if (meta->is_expert()) {
        this->expert_layers[meta->layer_id]->add_batch(tensor, meta);
    } else {
        ASSERT_MSG(false, "Invalid batch metadata");
    }
}

std::vector<int> UnifiedDefraggingLayerScheduler::get_pool_snapshot() {
    std::vector<int> snapshot(this->num_layers, 0);
    for (int i = 0; i < this->num_attn_layers; i++) {
        snapshot[i] = this->attn_layers[i]->get_num_tokens();
    }
    for (int i = 0; i < this->num_expert_layers; i++) {
        snapshot[i + this->num_attn_layers] =
            this->expert_layers[i]->get_num_tokens();
    }
    return snapshot;
}

TokenBatch UnifiedDefraggingLayerScheduler::get_batch_from_layer(int layer_id) {
    if (layer_id < 0 || layer_id >= this->num_layers) {
        return TokenBatch {};
    }
    auto batches = this->layers[layer_id]->get_all_batches();
    auto batch = TokenBatch::merge(batches);
    return batch;
}

TokenBatch UnifiedDefraggingLayerScheduler::get_batch_from_layer_restricted(
    int layer_id, int token_threshold) {
    if (layer_id < 0 || layer_id >= this->num_layers) {
        return TokenBatch {};
    }
    if (token_threshold > 0 &&
        this->layers[layer_id]->get_num_tokens() > token_threshold) {
        auto batches =
            this->layers[layer_id]->get_batches_restricted(token_threshold);
        auto batch = TokenBatch::merge(batches);
        return batch;
    }
    return this->get_batch_from_layer(layer_id);
}

/*

    Legacy layer scheduler

*/

LegacyLayerScheduler::LegacyLayerScheduler(int n_layers): 
    LegacyLayerScheduler(n_layers, LayerScheduleType::FLFS) { }

LegacyLayerScheduler::LegacyLayerScheduler(int n_layers, LegacyLayerScheduler::LayerScheduleType type): 
    LegacyLayerScheduler(n_layers, type, 0) { }

LegacyLayerScheduler::LegacyLayerScheduler(int n_layers, LegacyLayerScheduler::LayerScheduleType type, int lookback_steps):
    LegacyLayerScheduler(n_layers, type, lookback_steps, 1) { }

LegacyLayerScheduler::LegacyLayerScheduler(int n_layers, LegacyLayerScheduler::LayerScheduleType type, int lookback_steps, int block_size): 
    n_layers(n_layers), type(type), lookback_steps(lookback_steps), block_size(block_size),
    num_tokens_in_layer(std::vector<int>(n_layers, 0)), num_batches_in_layer(std::vector<int>(n_layers, 0)) { 
    if (lookback_steps > 0) {
        history_tokens_in_layer = std::vector<std::queue<int>>(n_layers, std::queue<int>());
        sum_history_tokens_in_layer = std::vector<int>(n_layers, 0);
    }
}

void LegacyLayerScheduler::add_tokens_to_layer(int layer_id, int num_tokens) {
    this->num_tokens_in_layer[layer_id] += num_tokens;
    this->num_batches_in_layer[layer_id] += 1;
}

void LegacyLayerScheduler::step_end() {
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

int LegacyLayerScheduler::schedule() {
    int layer_id = -1;
    switch (this->type) {
        case LayerScheduleType::MBFS:
            layer_id = this->_schedule_mbfs();
            break;
        case LayerScheduleType::FLFS:
            layer_id = this->_schedule_flfs();
            break;
        case LayerScheduleType::MBFLFS:
            layer_id = this->_schedule_mbflfs();
            break;
        case LayerScheduleType::MBTFS:
            layer_id = this->_schedule_batches_tokens();
            break;
        default:
            throw std::runtime_error("Unknown schedule type.");
    }
    step_end();
    clean_layer_status(layer_id);
    return layer_id;
}

void LegacyLayerScheduler::set_schedule_type(std::string type) {
    if (type == "mbfs") {
        this->type = LegacyLayerScheduler::LayerScheduleType::MBFS;
    } else if (type == "bin") {
        this->type = LegacyLayerScheduler::LayerScheduleType::BIN;
    } else if (type == "flfs") {
        this->type = LegacyLayerScheduler::LayerScheduleType::FLFS;
    } else if (type == "mbflfs") {
        this->type = LegacyLayerScheduler::LayerScheduleType::MBFLFS;
    } else if (type == "mbtfs") {
        this->type = LegacyLayerScheduler::LayerScheduleType::MBTFS;
    } else {
        throw std::runtime_error(type + " schedule not implemented.");
    }
}

void LegacyLayerScheduler::set_block_size(int block_size) {
    this->block_size = block_size;
}

int LegacyLayerScheduler::_schedule_bin() {
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

int LegacyLayerScheduler::_schedule_mbfs() {
    int scheduled_layer_id = 0;
    for (int i = 0; i < n_layers; i++) {
        if (num_tokens_in_layer[i] > num_tokens_in_layer[scheduled_layer_id]) {
            scheduled_layer_id = i;
        }
    }
    return scheduled_layer_id;
}

int LegacyLayerScheduler::_schedule_flfs() {
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

int LegacyLayerScheduler::_schedule_mbflfs() {
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

int LegacyLayerScheduler::_schedule_batches_tokens() {
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
    LegacyLayerScheduler(n_layers),
    hold_steps(hold_steps), layer_status(std::vector<LayerStatus>(n_layers, LayerStatus::IDLE)),
    num_steps_to_hold(std::vector<int>(n_layers, 0)), ready_timestamp_ms(std::vector<long long>(n_layers, 0)) {
}

int AdvancedLayerScheduler::schedule() {
    static int max_wait_time_ms = 100;
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
            if (num_tokens_in_layer[layer_id] == 0) {
                scores[i] = .0f;
                continue;
            }
            float decay = weight_decay;
            float score = num_tokens_in_layer[layer_id];
            for (int j = 1; j < lookahead_steps; j++) {
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
    ASSERT (num_tokens_in_layer[layer_to_schedule] > 0);
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
    LegacyLayerScheduler(num_layers * num_groups), n_groups(num_groups) { 
    this->n_layers = num_layers;
}

GroupLayerScheduler::GroupLayerScheduler(int num_layers, int num_groups, int lookback_steps):
    LegacyLayerScheduler(num_layers * num_groups, LayerScheduleType::FLFS, lookback_steps), n_groups(num_groups) { 
    this->n_layers = num_layers;
}

void GroupLayerScheduler::add_tokens_to_layer(int layer_id, int group_id, int num_tokens) {
    int layer_group_id = get_layer_group_id(layer_id, group_id);
    add_tokens_to_layer(layer_group_id, num_tokens);
}

int GroupLayerScheduler::schedule() {
    std::vector<float> scores(n_layers * n_groups);
    for (int i = 0; i < n_layers; i++) {
        float lookahead_score = 0;
        float decay = weight_decay;
        for (int k = 1; k < lookahead_steps; k++) {
            int cur_layer = (i + k) % n_layers;
            int num_tokens_cur_layer = 0;
            float history_score = .0f;
            for (int j = 0; j < n_groups; j++) {
                int layer_group_id = get_layer_group_id(cur_layer, j);
                num_tokens_cur_layer += num_tokens_in_layer[layer_group_id];
                if (lookback_steps > 0 && sum_history_tokens_in_layer[layer_group_id] > 0) {
                    history_score += sum_history_tokens_in_layer[layer_group_id] / history_tokens_in_layer[layer_group_id].size();
                }
            }
            lookahead_score += num_tokens_cur_layer * decay / n_groups;
            lookahead_score += history_score * decay / n_groups;
            decay *= weight_decay;
        }

        for (int j = 0; j < n_groups; j++) {
            int layer_group_id = get_layer_group_id(i, j);
            if (num_tokens_in_layer[layer_group_id] > 0) {
                scores[layer_group_id] = lookahead_score + num_tokens_in_layer[layer_group_id];
            } else {
                scores[layer_group_id] = .0f;
            }
        }
    }
    auto max_iter = std::max_element(scores.begin(), scores.end());
    int layer_group_id = std::distance(scores.begin(), max_iter);
    step_end();
    clean_layer_status(layer_group_id);
    return layer_group_id;
}
