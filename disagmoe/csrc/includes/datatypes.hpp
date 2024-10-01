#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>

#include "nccl.h"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>

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

    template<class Archive>
    void serialize(Archive &archive) {
        archive(req_id, exp_id, first_attn_id, prefill_pos);
    }

    friend std::ostream& operator<<(std::ostream &out, const TokenMetadata& token) {
        out << "TokenMetadata{req_id=" << token.req_id << ", "
            << "exp_id=" << token.exp_id << ", "
            << "first_attn_id=" << token.first_attn_id << ", "
            << "prefill_pos=" << token.prefill_pos << "}";
        return out;
    }
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
        auto shape = this->shape;
        shape[0] = r - l;
        auto infos = std::vector<TokenMetadata>(
            this->infos.begin() + l, 
            this->infos.begin() + r);
        return Metadata{
            shape, this->dtype, this->layer_id, infos, this->prompt_lens
        };
    }

    inline Metadata at(const std::vector<int>& ids) const {
        auto shape = this->shape;
        shape[0] = ids.size();
        auto infos = std::vector<TokenMetadata>(ids.size());
        for (size_t i = 0; i < ids.size(); i ++)
            infos[i] = this->infos[ids[i]];
        return Metadata{
            shape, this->dtype, this->layer_id, infos, this->prompt_lens
        };
    }

    template<class Archive>
    void serialize(Archive &archive) {
        archive(shape, dtype, layer_id, infos, prompt_lens);
    }

    size_t size() {
        return sizeof(this);
    }

    friend std::ostream& operator<<(std::ostream &out, const Metadata& meta) {
        out << "Metadata{";
        {
            out << "shape=(" << meta.shape[0];
            for (size_t i = 1; i < meta.shape.size(); i ++)
                out << ", " << meta.shape[i];
            out << "), ";

            out << "layer_id=" << meta.layer_id << ", ";

            out << "infos={";
            if (meta.infos.size() > 0) {
                out << meta.infos[0];
                for (size_t i = 1; i < meta.infos.size(); i ++)
                    out << ", " << meta.infos[i];
            }
            out << "}";
        }
        out << "}";
        return out;
    }
};

typedef std::shared_ptr<Metadata> metadata_t;

struct TensorBatch {
    uintptr_t data;
    metadata_t metadata;
};