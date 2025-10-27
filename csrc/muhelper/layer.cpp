#include "layer.h"

UnifiedLayer::UnifiedLayer(LayerType layer_type, int layer_id): 
    layer_type(layer_type), layer_id(layer_id), expert_id(-1), num_tokens(0), num_batches(0) {}

UnifiedLayer::UnifiedLayer(LayerType layer_type, int layer_id, int expert_id): 
    layer_type(layer_type), layer_id(layer_id), expert_id(expert_id), num_tokens(0), num_batches(0) {}

UnifiedLayer UnifiedLayer::create_attention_layer(int layer_id) {
    return UnifiedLayer(LayerType::ATTENTION, layer_id);
}

UnifiedLayer UnifiedLayer::create_expert_layer(int layer_id, int expert_id) {
    return UnifiedLayer(LayerType::EXPERT, layer_id, expert_id);
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
    result.swap(this->batch_queue);
    this->num_tokens = 0;
    this->num_batches = 0;
    return result;
}