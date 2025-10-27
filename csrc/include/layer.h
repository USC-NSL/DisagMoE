#pragma once

#ifndef LAYER_H_
#define LAYER_H_

#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
#include <torch/torch.h>

enum class LayerType { ATTENTION, EXPERT };

class UnifiedLayer {

private:
    LayerType layer_type;
    int layer_id;
    int expert_id; // >= 0 if is an individual expert
    int num_tokens;
    int num_batches;

    std::vector<TokenBatch> batch_queue;

public:
    UnifiedLayer(LayerType layer_type, int layer_id);

    UnifiedLayer(LayerType layer_type, int layer_id, int expert_id);

    static UnifiedLayer create_attention_layer(int layer_id);

    static UnifiedLayer create_expert_layer(int layer_id, int expert_id);

    inline bool is_attention() const { return layer_type == LayerType::ATTENTION; }

    inline bool is_expert() const { return layer_type == LayerType::EXPERT; }

    inline int get_layer_id() const { return layer_id; }

    inline int get_num_tokens() const { return num_tokens; }

    inline int get_num_batches() const { return num_batches; }

    void add_batch(const TokenBatch &batch);

    void add_batch(torch::Tensor tensor, const batch_metadata_t &meta);

    std::vector<TokenBatch> get_all_batches();

};

#endif