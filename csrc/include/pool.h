#pragma once

#ifndef POOL_H_
#define POOL_H_

#include "muhelper.h"
#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
#include "layer.h"
#include <string>

class UnifiedPool: public MuPool {

private:

    int top_k;

    int attn_schedule_token_threshold{-1};

    int expert_schedule_token_threshold{-1};

    std::vector<TokenTopKPool> topk_pools;

    unified_layer_scheduler_t layer_scheduler;

    void process_attn_batch_topk(torch::Tensor tensor, batch_metadata_t &meta);

    void process_attn_batch(torch::Tensor tensor, batch_metadata_t &meta);
    
    void process_expert_batch(torch::Tensor tensor, batch_metadata_t &meta);

    void process_batch(torch::Tensor tensor, batch_metadata_t &meta) override;

public:

    UnifiedPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels,
        int num_groups,
        int top_k,
        const std::string &unified_scheduler_type,
        float defrag_weight_decay,
        int defrag_lookahead_steps,
        int defrag_lookback_steps
    );

    void set_attn_schedule_token_threshold(int token_threshold);

    void set_expert_schedule_token_threshold(int token_threshold);

    TokenBatch get_batch_from_layer(int layer_id) override;

    std::vector<int> get_pool_snapshot() override;

    UnifiedLayerSchedulerBase::ScheduleResult schedule_with_snapshot();

    std::shared_ptr<LayerSchedulerBase> get_layer_scheduler();

    std::vector<int> get_topk_pool_snapshot();
};

using unified_pool_t = std::shared_ptr<UnifiedPool>;

#endif