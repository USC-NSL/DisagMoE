#include "pool.h"

UnifiedPool::UnifiedPool(
    std::vector<int> layer_ids,
    int device_id,
    std::vector<Channel_t> channels,
    int num_groups,
    int top_k,
    const std::string &unified_scheduler_type,
    float defrag_weight_decay,
    int defrag_lookahead_steps,
    int defrag_lookback_steps
):
    MuPool(layer_ids, device_id, channels, num_groups), top_k(top_k) {
    int num_layers = static_cast<int>(layer_ids.size());
    int num_attn_layers = num_layers + 1;
    int num_expert_layers = num_layers;
    if (unified_scheduler_type == "defrag") {
        this->layer_scheduler = std::make_shared<UnifiedDefraggingLayerScheduler>(
            num_attn_layers,
            num_expert_layers,
            top_k,
            defrag_lookback_steps,
            defrag_lookahead_steps,
            defrag_weight_decay
        );
    } else {
        this->layer_scheduler = std::make_shared<UnifiedLayerScheduler>(
            num_attn_layers,
            num_expert_layers,
            top_k
        );
    }
    if (top_k > 1) {
        this->topk_pools = std::vector<TokenTopKPool>(num_attn_layers, TokenTopKPool(top_k));
    }
}

void UnifiedPool::set_attn_schedule_token_threshold(int token_threshold) {
    this->attn_schedule_token_threshold = token_threshold;
}

void UnifiedPool::set_expert_schedule_token_threshold(int token_threshold) {
    this->expert_schedule_token_threshold = token_threshold;
}

void UnifiedPool::process_attn_batch_topk(torch::Tensor tensor, batch_metadata_t &meta) {
    this->topk_pools[meta->layer_id].put_batch((TokenBatch) {tensor, meta});
    auto ready_tokens = this->topk_pools[meta->layer_id].fetch_ready_tokens();
    int batched_tokens = ready_tokens.size();
    if (batched_tokens == 0) {
        return;
    }
    // std::lock_guard<std::mutex> lock(this->batch_mutex);
    this->layer_scheduler->attn_add_tokens(meta->layer_id, ready_tokens);
}

void UnifiedPool::process_attn_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    meta->batch_tag = BatchTag::ATTENTION;

    int num_prefill_seqs = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0; 

    for (int i = 0; i < meta->req_ids.size(); i ++) {
        if (meta->init_prefill_lens[i] != -1) {
            num_prefill_tokens ++;
            num_prefill_seqs ++;
        } else {
            num_decode_tokens ++;
        }
    }

    meta->num_prefill_seqs = num_prefill_seqs;
    meta->num_prefill_tokens = num_prefill_tokens;
    meta->num_decode_tokens = num_decode_tokens;

    // std::lock_guard<std::mutex> lock(this->batch_mutex);
    this->layer_scheduler->add_batch(tensor, meta);
}

void UnifiedPool::process_expert_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    meta->batch_tag = BatchTag::EXPERT;
    if (this->num_groups > 1) {
        throw std::runtime_error("Expert pool does not support multiple groups at this time");
    } else {
        // std::lock_guard<std::mutex> lock(this->batch_mutex);
        this->layer_scheduler->add_batch(tensor, meta);
    }
}

void UnifiedPool::process_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    if (meta->is_tokenizer()) {
        this->process_attn_batch(tensor, meta);
    } else if (meta->is_expert()) {
        if (this->top_k > 1 && meta->layer_id > 0) {
            this->process_attn_batch_topk(tensor, meta);
        } else {
            this->process_attn_batch(tensor, meta);
        }
    } else if (meta->is_attention()) {
        this->process_expert_batch(tensor, meta);
    } else {
        ASSERT_MSG(false, "Invalid batch metadata");
    }
}

TokenBatch UnifiedPool::get_batch_from_layer(int layer_id) {
    // std::lock_guard<std::mutex> lock(this->batch_mutex);

    int token_threshold = -1;
    if (this->layer_scheduler->is_attn_layer(layer_id)) {
        token_threshold = this->attn_schedule_token_threshold;
    } else {
        token_threshold = this->expert_schedule_token_threshold;
    }
    return this->layer_scheduler->get_batch_from_layer_restricted(layer_id, token_threshold);
}

std::shared_ptr<LayerSchedulerBase> UnifiedPool::get_layer_scheduler() {
    return std::dynamic_pointer_cast<LayerSchedulerBase>(this->layer_scheduler);
}

std::vector<int> UnifiedPool::get_pool_snapshot() {
    return this->layer_scheduler->get_pool_snapshot();
}

UnifiedLayerSchedulerBase::ScheduleResult UnifiedPool::schedule_with_snapshot() {
    return this->layer_scheduler->schedule_with_snapshot();
}

std::vector<int> UnifiedPool::get_topk_pool_snapshot() {
    ASSERT_MSG(this->top_k > 1, "Top-k pool snapshot is only supported for top-k > 1");
    std::vector<int> snapshot(this->topk_pools.size(), 0);
    for (int i = 0; i < this->topk_pools.size(); i++) {
        snapshot[i] = this->topk_pools[i].get_pool_size();
    }
    return snapshot;
}