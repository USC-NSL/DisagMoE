#pragma once

#ifndef POOL_H_
#define POOL_H_

#include "muhelper.h"
#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
#include "layer.h"

class UnifiedPool: public MuPool {

private:

    int top_k;

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
        int num_groups = 1,
        int top_k = 1
    );

    TokenBatch get_batch_from_layer(int layer_id) override;

    std::vector<int> get_pool_snapshot() override;

    std::shared_ptr<LayerSchedulerBase> get_layer_scheduler();

    std::vector<int> get_topk_pool_snapshot();
};

using unified_pool_t = std::shared_ptr<UnifiedPool>;

#endif