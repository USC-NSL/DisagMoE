#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "nccl.h"

struct ChannelInfo {
    std::vector<int> expert_ids;
    std::vector<int> attn_layer_ids;

    inline bool is_sampler_channel() {
        return expert_ids.empty() && attn_layer_ids.empty();
    }
};

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

    inline size_t num_element() const {
        size_t res = 1;
        for (size_t s: this->shape)
            res *= s;
        return res;
    }

    inline size_t num_tokens() const {
        return shape[0];
    }

    inline ncclDataType_t get_nccl_datatype() const {
        return ncclFloat16;
    }

    inline size_t get_datatype_size() const {
        return 2; // fp16
    }

    inline Metadata slice(int l, int r) {
        shape = this->shape;
        shape[0] = r - l;
        auto infos = std::vector<TokenMetadata>(
            this->infos.begin() + l, 
            this->infos.begin() + r);
        return Metadata{
            shape, this->dtype, this->layer_id, infos, this->prompt_lens
        };
    }
};

typedef std::shared_ptr<Metadata> metadata_t;

struct TensorBatch {
    uintptr_t data;
    metadata_t metadata;
};