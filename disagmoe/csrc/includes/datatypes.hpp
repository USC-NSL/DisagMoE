#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cassert>
#include <algorithm>

#include "nccl.h"
#include "cuda_utils.h"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>

struct ChannelInfo {
    std::vector<int> expert_ids;
    std::vector<int> attn_layer_ids;

    ChannelInfo() {}
    ChannelInfo(const std::vector<int> &expert_ids,
                const std::vector<int> &attn_layer_ids):
                expert_ids(expert_ids), attn_layer_ids(attn_layer_ids) 
    {}

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

struct Metadata;

typedef std::shared_ptr<Metadata> metadata_t;

struct Metadata {
    std::vector<size_t> shape;
    std::string dtype;

    int layer_id;
    std::vector<TokenMetadata> infos;
    std::map<int, int> prompt_lens;

    // Metadata() {}
    // Metadata(std::vector<size_t> shape, 
    //          std::string dtype,
    //          int layer_id,
    //          std::vector<TokenMetadata> infos,
    //          std::map<int, int> prompt_lens = {}): 
    //             shape(shape), dtype(dtype), layer_id(layer_id), 
    //             infos(infos), prompt_lens(prompt_lens) 
    //         {}

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

    static metadata_t merge(const std::vector<metadata_t> &metas) {
        assert(metas.size() > 0);
        std::vector<size_t> shape = metas[0]->shape;
        auto dtype = metas[0]->dtype;
        auto layer_id = metas[0]->layer_id;
        std::vector<TokenMetadata> infos;
        std::map<int, int> prompt_lens;

        for (size_t i = 1; i < metas.size(); i ++) {
            auto meta = metas[i];
            shape[0] += meta->shape[0];
            for (auto info: meta->infos)
                infos.push_back(info);
            for (auto [k, v]: meta->prompt_lens)
                prompt_lens[k] = v;
        }

        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, infos, prompt_lens
        });
    }

    void step_layer() {
        this->layer_id ++;
    }

    void update_exp_ids(const std::vector<int> &new_exp_ids, bool required_sort) {
        assert(new_exp_ids.size() == infos.size());
        for (int i = 0; i < infos.size(); i ++) {
            infos[i].exp_id = new_exp_ids[i];
        }
        if (required_sort)
            std::sort(infos.begin(), infos.end(),
                [](const TokenMetadata &l, const TokenMetadata &r) {
                    return l.exp_id < r.exp_id;
                });
    }
};

struct TensorBatch {
    uintptr_t data;
    metadata_t metadata;

    static TensorBatch merge(const std::vector<TensorBatch>& batches) {
        std::vector<metadata_t> metas(batches.size());
        for (size_t i = 0; i < batches.size(); i ++) {
            metas[i] = batches[i].metadata;
        }
        auto meta = Metadata::merge(metas);

        auto dtype = meta->get_datatype_size();
        
        uintptr_t buf = alloc_cuda_tensor(meta->num_element(), 0);
        
        uintptr_t ptr = buf;
        for (auto &batch: batches) {
            auto size = batch.metadata->num_element() * dtype;
            cudaMemcpy((void*) ptr, (void*) batch.data, size, 
                cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            ptr += size;
        }

        return TensorBatch {buf, meta};
    }
};

struct AttentionBatchMetadata;

typedef std::shared_ptr<AttentionBatchMetadata> attn_metadata_t;

struct AttentionBatchMetadata {
    int layer_id;
    std::vector<size_t> shape;
    std::string dtype;

    int num_prefill_seqs;
    int num_prefill_tokens;

    int num_decode_tokens; // equals to num_decode_seqs as each decode seq has query length of 1

    std::vector<int> seq_ids; // per seq, length of (num_prefill_seqs + num_decode_tokens)

    /*
    |----------seq_len----------|
    | last chunk  | this chunk  |
    |-------------|-------------|
    |-context_len-|
                  |--query_len--|

    context = seq - query
    */

    std::vector<int> prefill_seq_len; // per perfill seq, length of (num_prefill_seqs)
    std::vector<int> prefill_query_len; // per perfill seq, length of (num_prefill_seqs)

    std::vector<uint8_t> expert_ids; // optional, per token, length of (num_prefill_tokens + num_decode_tokens)

    // place holder for first attention id.
    // std::vector<uint8_t> first_attn_ids; 

    int prefill_data_size() {
        return num_prefill_tokens * shape[1] * 2; // assume only bf16 or fp16
    }

    int decode_data_size() {
        return num_decode_tokens * shape[1] * 2; // assume only bf16 or fp16
    }

    static attn_metadata_t merge(const std::vector<attn_metadata_t>& batches) {
        int new_prefills_seqs = 0;
        int new_prefill_tokens = 0;
        int new_decode_tokens = 0;

        std::vector<int> new_seq_ids{};
        std::vector<int> new_prefill_seq_len{};
        std::vector<int> new_prefill_query_len{};

        for (auto &batch: batches) {
            new_prefills_seqs += batch->num_prefill_seqs;
            new_prefill_tokens += batch->num_prefill_tokens;
            new_decode_tokens += batch->num_decode_tokens;

            for (int i = 0; i < batch->num_prefill_seqs; i++) {
                new_seq_ids.emplace_back(batch->seq_ids[i]);
                new_prefill_seq_len.emplace_back(batch->prefill_seq_len[i]);
                new_prefill_query_len.emplace_back(batch->prefill_query_len[i]);
            }
        }

        for (auto &batch: batches) {
            for (int i = batch->num_prefill_seqs; i < batch->num_prefill_seqs + batch->num_decode_tokens; i++) {
                new_seq_ids.emplace_back(batch->seq_ids[i]);
            }
        }

        return std::make_shared<AttentionBatchMetadata> (
            AttentionBatchMetadata {
                batches[0]->layer_id,
                batches[0]->shape,
                batches[0]->dtype,
                new_prefills_seqs,
                new_prefill_tokens,
                new_decode_tokens,
                new_seq_ids,
                new_prefill_seq_len,
                new_prefill_query_len
            }
        );
    }
};


struct AttentionBatch {
    uintptr_t data;
    attn_metadata_t metadata;

    static AttentionBatch merge(const std::vector<AttentionBatch>& batches) {
        std::vector<attn_metadata_t> metas(batches.size());
        for (size_t i = 0; i < batches.size(); i ++) {
            metas[i] = batches[i].metadata;
        }
        auto meta = AttentionBatchMetadata::merge(metas);

        int prefill_data_size = meta->prefill_data_size();
        int decode_data_size = meta->decode_data_size();
        
        uintptr_t buf = alloc_cuda_tensor(prefill_data_size + decode_data_size, 0);
        
        void* prefill_ptr = (void *)buf;
        void* decode_ptr = prefill_ptr + prefill_data_size;

        for (auto &batch: batches) {
            int prefill_copy_size = batch.metadata->prefill_data_size();
            int decode_copy_size = batch.metadata->decode_data_size();
            cudaMemcpy(prefill_ptr, (void *)batch.data, prefill_copy_size, 
                cudaMemcpyKind::cudaMemcpyDeviceToDevice);

            cudaMemcpy(decode_ptr, (void *)batch.data + prefill_copy_size, decode_copy_size,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            prefill_ptr += prefill_copy_size;
            decode_ptr += decode_copy_size;
        }

        return AttentionBatch {buf, meta};
    }
};