#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <ctime>

#include "nccl.h"
#include "cuda_utils.h"
#include "logging.h"
#include "constants.h"
#include "permute.h"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include <torch/torch.h>

template<class T>
inline std::vector<T> slice_vector(const std::vector<T> &a, int l, int r) {
    std::vector<T> res;
    for (int i = l; i < r; i ++)    
        res.emplace_back(std::move(a[i]));
    return res;
}


template<class T>
inline void extend(std::vector<T> &a, const std::vector<T> &other) {
    for (const T &v: other)
        a.emplace_back(v);
}

// first == layer_id, second == expert_id
#define ExpertId std::pair<int, int>


struct ChannelInfo {
    std::vector<ExpertId> expert_ids;
    std::vector<int> attn_layer_ids;

    ChannelInfo() {}
    ChannelInfo(const std::vector<ExpertId> &expert_ids,
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
    int prefill_pos;

    template<class Archive>
    void serialize(Archive &archive) {
        archive(req_id, exp_id, prefill_pos);
    }

    friend std::ostream& operator<<(std::ostream &out, const TokenMetadata& token) {
        out << "TokenMetadata{req_id=" << token.req_id << ", "
            << "exp_id=" << token.exp_id << ", "
            // << "first_attn_id=" << token.first_attn_id << ", "
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
    std::vector<int> req_ids;
    std::vector<int> exp_ids;
    // std::vector<int> first_attn_ids;
    std::vector<int> prefill_poss;
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

    inline size_t token_hidden_dim() const {
        return shape[1];
    }

    inline ncclDataType_t get_nccl_datatype() const {
        return ncclBfloat16;
    }

    constexpr size_t get_datatype_size() const {
        return 2; // bf16
    }

    inline Metadata slice(int l, int r) {
        auto shape = this->shape;
        shape[0] = r - l;
        return Metadata{
            shape, this->dtype, this->layer_id, 
            slice_vector(req_ids, l, r), 
            slice_vector(exp_ids, l, r),
            slice_vector(prefill_poss, l, r),
            this->prompt_lens
        };
    }

    inline Metadata at(const std::vector<int>& ids) const {
        auto shape = this->shape;
        shape[0] = ids.size();
        std::vector<int> req_ids_(shape[0]), exp_ids_(shape[0]), prefill_poss_(shape[0]);
        for (size_t i = 0; i < ids.size(); i ++) {
            ASSERT (ids[i] < this->req_ids.size());
            req_ids_[i] = req_ids[ids[i]];
            exp_ids_[i] = exp_ids[ids[i]];
            prefill_poss_[i] = prefill_poss[ids[i]];
        }
        return Metadata{
            shape, this->dtype, this->layer_id, req_ids_, exp_ids_, prefill_poss_, this->prompt_lens
        };
    }

    template<class Archive>
    void serialize(Archive &archive) {
        archive(shape, dtype, layer_id, req_ids, exp_ids, prefill_poss, prompt_lens);
    }

    size_t size() {
        return sizeof(this);
    }

    std::vector<int> get_expert_batch_sizes(int n_expert) {
        ASSERT(n_expert > 0);
        std::vector<int> batches(n_expert, 0);
        for (int eid: exp_ids)
            batches[eid] += 1;
        return batches;
    }

    TokenMetadata info_at(int i) const {
        return TokenMetadata {req_ids[i], exp_ids[i], prefill_poss[i]};
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
            if (meta.req_ids.size() > 0) {
                out << meta.info_at(0);
                for (size_t i = 1; i < meta.req_ids.size(); i ++)
                    out << ", " << meta.info_at(i);
            }
            out << "}";
        }
        out << "}";
        return out;
    }

    static metadata_t merge(const std::vector<metadata_t> &metas) {
        ASSERT(metas.size() > 0);
        std::vector<size_t> shape = metas[0]->shape;
        auto dtype = metas[0]->dtype;
        auto layer_id = metas[0]->layer_id;

        int total_tokens = 0;
        for (auto &meta: metas) {
            total_tokens += meta->num_tokens();
        }
        shape[0] = total_tokens;
        std::vector<int> req_ids, exp_ids, prefill_poss;
        
        req_ids.reserve(total_tokens);
        exp_ids.reserve(total_tokens);
        prefill_poss.reserve(total_tokens);

        std::map<int, int> prompt_lens;

        for (size_t i = 0; i < metas.size(); i ++) {
            auto meta = metas[i];
            extend(req_ids, meta->req_ids);
            extend(exp_ids, meta->exp_ids);
            extend(prefill_poss, meta->prefill_poss);
            for (auto &[k, v]: meta->prompt_lens)
                prompt_lens[k] = v;
        }

        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids, exp_ids, prefill_poss, prompt_lens
        });
    }

    void step_layer() {
        this->layer_id ++;
    }

    void update_exp_ids(const std::vector<int> &new_exp_ids,
                        const std::vector<int> &exp_mappings) {
        exp_ids = new_exp_ids;
        permute_token_infos(exp_mappings);
    }

    void permute_token_infos(const std::vector<int> &exp_mappings) {
        if (exp_mappings.empty())
            return;
        ASSERT (req_ids.size() == exp_mappings.size());
        std::vector<int> tmp(exp_mappings.size());
        #define MOVE(a) { \
            for (int i = 0; i < exp_mappings.size(); i ++) {    \
                int j = exp_mappings[i];                        \
                ASSERT(0 <= j && j < tmp.size());               \
                tmp[j] = a[i];                                  \
            }                                                   \
            a = tmp;                                            \
        }
        MOVE(exp_ids);
        MOVE(req_ids);
        MOVE(prefill_poss);
        #undef MOVE
    }

    std::vector<int> sort_by_prefill_order() {
        // TODO: deal with multiple prefill tokens
        std::vector<int> rank(req_ids.size()), mapping(req_ids.size());
        for (int i = 0; i < req_ids.size(); i ++)
            rank[i] = i;
        std::sort(
            rank.begin(), rank.end(),
            [&](const int i, const int j) {
                return (prefill_poss[i] < 0 || prefill_poss[j] < 0) ?
                    prefill_poss[i] > prefill_poss[j] :
                    (prefill_poss[i] == prefill_poss[j] ? 
                        req_ids[i] < req_ids[j] :
                        prefill_poss[i] < prefill_poss[j]);
            }
        );
        for (int i = 0; i < req_ids.size(); i ++)
            mapping[rank[i]] = i;
        update_exp_ids({}, mapping);
        return mapping;
    }
};

struct TensorBatch {
    torch::Tensor data;
    metadata_t metadata;

    static TensorBatch merge(const std::vector<TensorBatch>& batches) {
        AUTO_TX_RANGE;

        if (batches.empty()) {
            return TensorBatch {};
        }
        cudaStream_t stream = get_current_torch_stream();

        std::vector<metadata_t> metas(batches.size());
        for (size_t i = 0; i < batches.size(); i ++) {
            metas[i] = batches[i].metadata;
        }
        auto meta = Metadata::merge(metas);

        torch::Tensor tensor = torch::empty(
            {meta->num_tokens(), meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );

        // Option 1. use cudaMemcpy
        // auto dtype = meta->get_datatype_size();
        // void* buf = tensor.data_ptr();
        // void* ptr = buf;
        // for (auto &batch: batches) {
        //     auto size = batch.metadata->num_element() * dtype;
        //     cudaMemcpy((void*) ptr, (void*) batch.data.data_ptr(), size, 
        //         cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        //     ptr += size;
        // }

        // Option 2. use gather cuda kernel

        std::vector<uintptr_t> srcs(meta->num_tokens());

        int idx = 0;
        int hidden_size_bytes = meta->token_hidden_dim() * meta->get_datatype_size();
        for (auto &batch: batches) {
            uintptr_t cur_ptr = (uintptr_t) batch.data.data_ptr();
            for (int i = 0; i < batch.metadata->num_tokens(); i ++) {
                srcs[idx] = cur_ptr;
                cur_ptr += hidden_size_bytes;
                idx ++;
            }
        }

        gather_tokens_cuda(tensor, srcs.data(), meta->num_tokens(), meta->token_hidden_dim(), stream);

        return TensorBatch {tensor, meta};
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

    constexpr int get_datatype_size() const {
        return 2; // bf16
    }

    inline int num_tokens() const {
        return shape[0];
    }

    inline int token_hidden_dim() const {
        return shape[1];
    }

    inline int prefill_data_size() const {
        return num_prefill_tokens * shape[1] * get_datatype_size(); 
    }

    inline int decode_data_size() const {
        return num_decode_tokens * shape[1] * get_datatype_size();
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
        auto new_shape = batches[0]->shape;
        for (int i = 1; i < batches.size(); i ++)
            new_shape[0] += batches[i]->shape[0];

        return std::make_shared<AttentionBatchMetadata> (
            AttentionBatchMetadata {
                batches[0]->layer_id,
                new_shape,
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

    metadata_t to_metadata() {
        auto shape = this->shape;
        auto dtype = this->dtype;
        auto layer_id = this->layer_id;
        std::vector<int> req_ids_;
        std::vector<int> prefill_poss_;

        // DMOE_LOG(INFO) << "To metadata, seq_ids: ";
        // for (int i = 0; i < num_prefill_seqs + num_decode_tokens; i ++)
        //     std::cout << seq_ids[i] << " ";
        // std::cout << LEND;
        
        for (int i = 0; i < num_prefill_seqs; i ++) {
            // TODO(hogura|20241014): modify to chunked prefill
            for (int j = 0; j < prefill_seq_len[i]; j ++) {
                req_ids_.push_back(seq_ids[i]);
                prefill_poss_.push_back(j);
                // TODO(hogura|20241014): add attention replica
            }
        }

        for (int i = 0; i < num_decode_tokens; i ++) {
            req_ids_.push_back(seq_ids[num_prefill_seqs + i]);
            prefill_poss_.push_back(-1);
        }
        
        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids_, {}, prefill_poss_, {}
        });
    }
};


struct AttentionBatch {
    torch::Tensor data;
    attn_metadata_t metadata;

    static AttentionBatch merge(const std::vector<AttentionBatch>& batches) {
        AUTO_TX_RANGE;
        
        if (batches.empty()) {
            return AttentionBatch {};
        }
        at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(true, -1);
        cudaStream_t stream = c10_stream.stream();
        at::cuda::CUDAStreamGuard guard(c10_stream);

        std::vector<attn_metadata_t> metas(batches.size());
        for (size_t i = 0; i < batches.size(); i ++) {
            metas[i] = batches[i].metadata;
        }
        auto meta = AttentionBatchMetadata::merge(metas);

        int prefill_data_size = meta->prefill_data_size();
        int decode_data_size = meta->decode_data_size();

        torch::Tensor tensor = torch::empty(
            {meta->num_tokens(), meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );

        // Option 1. use cudaMemcpy
        // void * buf = tensor.data_ptr();
        
        // uintptr_t buf = alloc_cuda_tensor((prefill_data_size + decode_data_size) / meta->get_datatype_size(), 0, sizeof(short), stream);
        
        // void* prefill_ptr = (void *)buf;
        // void* decode_ptr = prefill_ptr + prefill_data_size;
        // for (auto &batch: batches) {
        //     int prefill_copy_size = batch.metadata->prefill_data_size();
        //     int decode_copy_size = batch.metadata->decode_data_size();
        //     cudaMemcpyAsync(prefill_ptr, (void *)batch.data, prefill_copy_size, 
        //         cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream);

        //     cudaMemcpyAsync(decode_ptr, (void *)batch.data + prefill_copy_size, decode_copy_size,
        //         cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream);
        //     prefill_ptr += prefill_copy_size;
        //     decode_ptr += decode_copy_size;
        //     free_cuda_tensor((void *) batch.data, stream);
        // }
        
        // cudaStreamSynchronize(stream);

        // Option 2. use gather cuda kernel
        int prefill_idx = 0;
        int decode_idx = meta->num_prefill_tokens;

        std::vector<uintptr_t> src_ptrs(meta->num_tokens());
        const int hidden_size_byte = meta->token_hidden_dim() * tensor.element_size();

        for (auto &batch: batches) {
            uintptr_t cur_data_ptr = (uintptr_t) batch.data.data_ptr();

            for (int i = 0; i < batch.metadata->num_prefill_tokens; i ++) {
                src_ptrs[prefill_idx] = cur_data_ptr;
                cur_data_ptr += hidden_size_byte;
                prefill_idx ++;
            }

            for (int i = 0; i < batch.metadata->num_decode_tokens; i ++) {
                src_ptrs[decode_idx] = cur_data_ptr;
                cur_data_ptr += hidden_size_byte;
                decode_idx ++;
            }
        }
        DMOE_LOG(WARNING) << "start gather_tokens_cuda in stream " << stream << LEND;
        gather_tokens_cuda(tensor, src_ptrs.data(), meta->num_tokens(), meta->token_hidden_dim(), stream);
        DMOE_LOG(WARNING) << "end gather_tokens_cuda" << LEND;

        return AttentionBatch {tensor, meta};
    }
};

struct SloStat {
    clock_t t_prefill;  // time to all finished prefill tokens
    clock_t t_decode;   // time to all finished decode tokens

    std::vector<clock_t> t_tokens;
};

struct ParallelConfig {
    int tp = 1;
    int ep = 1;
    int n_exp_per_rank = 1;

    ParallelConfig(int tp, int ep, int n_exp_per_rank): 
        tp(tp), ep(ep), n_exp_per_rank(n_exp_per_rank) {}
};