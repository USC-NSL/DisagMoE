#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "nccl.h"

struct TokenMetadata {
    int req_id;
    int exp_id;
    int first_attn_id;
    int prefill_pos;
};

struct Metadata {
    std::vector<size_t> shape;
    std::string dtype;

    int layer_id;
    std::vector<TokenMetadata> infos;
    std::map<int, int> prompt_lens;

    inline size_t get_num_element() const {
        size_t res = 1;
        for (size_t s: this->shape)
            res *= s;
        return res;
    }

    inline ncclDataType_t get_nccl_datatype() const {
        return ncclFloat16;
    }
};

typedef std::shared_ptr<Metadata> metadata_t;

struct TensorBatch {
    void* data;
    metadata_t metadata;
};