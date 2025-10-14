#pragma once

#ifndef LAYER_H_
#define LAYER_H_

#include "datatypes.hpp"
#include <torch/torch.h>

class Layer {
private:
    int layer_id;
    int num_tokens;
    int num_batches;

public:
    Layer(int layer_id): layer_id(layer_id) {}

    int get_layer_id() const { return layer_id; }
    int get_num_tokens() const { return num_tokens; }
    int get_num_batches() const { return num_batches; }

    virtual void clear_layer() {
        this->num_tokens = 0;
        this->num_batches = 0;
    }

    virtual void add_batch(torch::Tensor tensor, metadata_t &meta) = 0;

    virtual void get_batch() = 0;

};

#endif