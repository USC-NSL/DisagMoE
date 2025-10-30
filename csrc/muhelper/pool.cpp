#include "pool.h"

UnifiedPool::UnifiedPool(
    std::vector<int> layer_ids, 
    int device_id, 
    std::vector<Channel_t> channels, 
    int num_groups
):
    MuPool(layer_ids, device_id, channels, num_groups) {
    this->layer_scheduler = std::make_shared<UnifiedLayerScheduler>(layer_ids.size() + 1);
}

void UnifiedPool::process_attn_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    // TODO: add topk support
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

    // TODO: deal with layer scheduler
    std::lock_guard<std::mutex> lock(this->batch_mutex);
    this->layer_scheduler->add_batch(tensor, meta);
}

void UnifiedPool::process_expert_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    meta->batch_tag = BatchTag::EXPERT;

    if (this->num_groups > 1) {
        throw std::runtime_error("Expert pool does not support multiple groups at this time");
    } else {
        std::lock_guard<std::mutex> lock(this->batch_mutex);
        this->layer_scheduler->add_batch(tensor, meta);
    }
}

void UnifiedPool::process_batch(torch::Tensor tensor, batch_metadata_t &meta) {
    if (meta->is_expert() || meta->is_tokenizer()) {
        this->process_attn_batch(tensor, meta);
    } else if (meta->is_attention()) {
        this->process_expert_batch(tensor, meta);
    } else {
        ASSERT_MSG(false, "Invalid batch metadata");
    }
}

std::vector<TokenBatch> UnifiedPool::get_batch_from_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(this->batch_mutex);
    return this->layer_scheduler->get_batch_from_layer(layer_id);
}

std::shared_ptr<LayerSchedulerBase> UnifiedPool::get_layer_scheduler() {
    return std::dynamic_pointer_cast<LayerSchedulerBase>(this->layer_scheduler);
}