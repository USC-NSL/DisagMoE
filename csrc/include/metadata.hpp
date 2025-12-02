#pragma once

#ifndef METADATA_H_
#define METADATA_H_

#include "datatypes.hpp"
#include "vector_utils.hpp"

#include <vector>
#include <string>
#include <optional>
#include <memory>
#include <algorithm>

#include "nccl.h"
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/optional.hpp>

constexpr int max_num_experts = 128;
constexpr int max_num_attn_dp_ranks = 32;

struct BatchMetadata;

typedef std::shared_ptr<BatchMetadata> batch_metadata_t;

struct BatchMetadata {
    BatchTag batch_tag;
    std::vector<size_t> shape;
    std::string dtype;

    int layer_id;
    std::vector<int> req_ids;
    std::vector<int> exp_ids;
    std::vector<float> topk_weights;

    std::vector<int> attn_dp_ranks;
    std::vector<int> init_prefill_lens; // positive for first decoding tokens, -1 for subsequence decoding tokens
    
    // Only used in attention batch.
    // Note: All metadata operations will ignore these optional fields.
    std::optional<int> num_prefill_seqs;
    std::optional<int> num_prefill_tokens;
    std::optional<int> num_decode_tokens;

    friend std::ostream& operator<<(std::ostream &out, const BatchMetadata &meta) {
        out << "BatchMetadata {";
        out << "num_tokens=" << meta.num_tokens() << ", ";
        out << "layer_id=" << meta.layer_id << ", ";
        for (int i = 0; i < meta.req_ids.size(); i++) {
            out << "token[i]: {";
            out << "req_id=" << meta.req_ids[i] << ", ";
            out << "attn_dp_rank=" << meta.attn_dp_ranks[i] << ", ";
            out << "init_prefill_len=" << meta.init_prefill_lens[i] << ", ";
            out << "}";
        }
        out << "}";
        return out;
    }

    template<class Archive>
    void serialize(Archive &archive) {
        archive(
            batch_tag, shape, dtype, layer_id, 
            req_ids, exp_ids, topk_weights, 
            attn_dp_ranks, init_prefill_lens,
            num_prefill_tokens, num_prefill_seqs, num_decode_tokens
        );
    }

    inline BatchTag get_batch_tag() const {
        return batch_tag;
    }
    
    inline bool is_attention() const {
        return batch_tag == BatchTag::ATTENTION;
    }

    inline bool attention_batch_safe_check() const {
        return num_prefill_tokens.has_value() && num_prefill_seqs.has_value() && num_decode_tokens.has_value() && 
               num_prefill_tokens.value() == num_prefill_seqs.value();
    }
    
    inline bool is_expert() const {
        return batch_tag == BatchTag::EXPERT;
    }

    inline bool is_tokenizer() const {
        return batch_tag == BatchTag::TOKENIZER;
    }

    inline int num_tokens() const {
        return shape[0];
    }

    inline int token_hidden_dim() const {
        return shape[1];
    }

    inline size_t num_element() const {
        size_t res = 1;
        for (size_t s: this->shape)
            res *= s;
        return res;
    }

    inline int get_datatype_size() const {
        return 2; // bf16
    }

    inline ncclDataType_t get_nccl_datatype() const {
        return ncclBfloat16;
    }

    inline int prefill_data_size() const {
        return num_prefill_tokens.value() * shape[1] * get_datatype_size(); 
    }

    inline int decode_data_size() const {
        return num_decode_tokens.value() * shape[1] * get_datatype_size();
    }

    inline int get_expert_id() const {
        // NOTE: this is only used in expert worker,
        //       caller must make sure all tokens in
        //       the batch have the same expert id
        return exp_ids[0];
    }


    inline void step_layer() {
        this->layer_id ++;
    }

    void set_finish_signal(const std::vector<int> &continue_ids) {
        for (auto &x: init_prefill_lens) {
            x = 0;
        }
        for (auto &x: continue_ids) {
            init_prefill_lens[x] = -1;
        }
    }

    std::vector<int> get_expert_batch_sizes(int n_expert) {
        ASSERT(n_expert > 0);
        std::vector<int> batches(n_expert, 0);
        for (int eid: exp_ids)
            batches[eid] += 1;
        return batches;
    }

    void get_expert_batch_sizes_cuda(int n_expert, const std::vector<int> &local_to_global_expert_rank, torch::Tensor tensor_cuda, uintptr_t stream_ptr) {
        AUTO_TX_RANGE;
        ASSERT(n_expert > 0);
        auto batch_sizes = get_expert_batch_sizes(n_expert);
        int64_t batches[MAX_N_EXPERTS];
        int m = local_to_global_expert_rank.size();
        for (int i = 0; i < m; i ++) {
            ASSERT(0 <= local_to_global_expert_rank[i] && local_to_global_expert_rank[i] < n_expert);
            batches[i] = batch_sizes[local_to_global_expert_rank[i]];
        }
        CUDACHECK(cudaMemcpyAsync(tensor_cuda.data_ptr(), batches, m * sizeof(int64_t), cudaMemcpyHostToDevice, cudaStream_t(stream_ptr)));
    }

    std::vector<int> get_token_expert_indices(int n_expert, const std::vector<int> &global_to_local_expert_rank) {
        std::vector<int> token_expert_indices(num_tokens());
        for (int i = 0; i < num_tokens(); i ++) {
            ASSERT(global_to_local_expert_rank[exp_ids[i]] != -1);
            token_expert_indices[i] = global_to_local_expert_rank[exp_ids[i]];
        }
        return token_expert_indices;
    }

    std::vector<int> get_finished_indices() {
        std::vector<int> finish_indices{};
        for (size_t i = 0; i < init_prefill_lens.size(); i ++) {
            if (init_prefill_lens[i] == 0) {
                finish_indices.emplace_back(i);
            }
        }
        return finish_indices;
    }

    BatchMetadata slice(int l, int r) {
        if (r - l == shape[0]) {
            return *this;
        }
        return BatchMetadata {
            batch_tag,
            std::vector<size_t> {r - l, shape[1]},
            dtype,
            layer_id,
            slice_vector(req_ids, l, r),
            slice_vector(exp_ids, l, r),
            slice_vector(topk_weights, l, r),
            slice_vector(attn_dp_ranks, l, r),
            slice_vector(init_prefill_lens, l, r),
        };
    }

    void permute_token_infos(const std::vector<int> &positions) {
        if (positions.empty())
            return;

        ASSERT (num_tokens() == positions.size());
        req_ids = permute_vector(req_ids, positions);
        exp_ids = permute_vector(exp_ids, positions);
        attn_dp_ranks = permute_vector(attn_dp_ranks, positions);
        init_prefill_lens = permute_vector(init_prefill_lens, positions);
        topk_weights = permute_vector(topk_weights, positions);
    }

    void duplicate_topk(int topk) {
        if (topk == 1) return;
        req_ids = duplicate_vector(req_ids, topk);
        init_prefill_lens = duplicate_vector(init_prefill_lens, topk);
        attn_dp_ranks = duplicate_vector(attn_dp_ranks, topk);
        topk_weights = duplicate_vector(topk_weights, topk);
        shape[0] *= topk;
    }

    std::vector<int> get_chunk_sizes() {
        // NOTE: token of same attention or expert must be consecutive
        std::vector<int> &index_vec = is_expert() ? exp_ids : attn_dp_ranks;
        std::vector<int> chunk_sizes;
        int last = 0;
        for (int i = 1; i < num_tokens(); i ++) {
            if (index_vec[i] != index_vec[i - 1]) {
                chunk_sizes.emplace_back(i - last);
                last = i;
            }
        }
        chunk_sizes.emplace_back(num_tokens() - last);
        return chunk_sizes;
    }

    std::vector<int> sort_by_attention() {
        // return value: corresponding positions after permutation. 
        //               e.g. positions[i] = j means tokens i should be at position j after permutation.

        std::vector<int> rank(req_ids.size()), positions(req_ids.size());
        for (int i = 0; i < req_ids.size(); i ++)
            rank[i] = i;
        std::sort(
            rank.begin(), rank.end(),
            [&](const int i, const int j) {
                if (attn_dp_ranks[i] != attn_dp_ranks[j]) {
                    return attn_dp_ranks[i] < attn_dp_ranks[j];
                }
                if (init_prefill_lens[i] == -1 || init_prefill_lens[j] == -1) {
                    return init_prefill_lens[i] > init_prefill_lens[j];
                }
                return req_ids[i] < req_ids[j];
            }
        );
        for (int i = 0; i < req_ids.size(); i ++)
            positions[rank[i]] = i;
        permute_token_infos(positions);
        return positions;
    }

    std::vector<int> sort_by_expert() {
        // return value: corresponding positions after permutation. 
        //               e.g. positions[i] = j means tokens i should be at position j after permutation.
        static std::array<int, max_num_experts> expert_cnts;
        expert_cnts.fill(0);
        std::vector<int> positions(num_tokens());
        for (auto &eid: exp_ids) {
            expert_cnts[eid] ++;
        }
        for (int i = 1; i < max_num_experts; i ++) {
            expert_cnts[i] += expert_cnts[i - 1];
        }
        std::vector<int> new_exp_ids(num_tokens());
        int first_expert_cnt = 0;
        for (int i = 0; i < num_tokens(); i ++) {
            if (exp_ids[i] == 0) {
                positions[i] = first_expert_cnt;
                first_expert_cnt ++;
            } else {
                int &prev_tokens = expert_cnts[exp_ids[i]-1];
                positions[i] = prev_tokens;
                prev_tokens ++;
            }
        }
        permute_token_infos(positions);
        return positions;
    }

    batch_metadata_t index_select(const std::vector<int> &indices) {
        int n = indices.size();
        return std::make_shared<BatchMetadata>(BatchMetadata {
            batch_tag,
            {n, shape[1]},
            dtype, 
            layer_id, 
            index_select_vector(req_ids, indices),
            index_select_vector(exp_ids, indices),
            index_select_vector(topk_weights, indices),
            index_select_vector(attn_dp_ranks, indices),
            index_select_vector(init_prefill_lens, indices)
        });
    }

    std::vector<BatchMetadata> split_with_sizes(const std::vector<int> &sizes);

    std::vector<BatchMetadata> split_by_indices(const std::vector<int> &positions);

    static batch_metadata_t merge_by_expert(const std::vector<batch_metadata_t> &metas, std::vector<int> &positions);

    static batch_metadata_t merge_by_attention(const std::vector<batch_metadata_t> &metas);

    static batch_metadata_t pack_topk_tokens(int layer_id, const std::vector<TokenTopKInfo>& tokens);
};


inline std::vector<BatchMetadata> BatchMetadata::split_with_sizes(const std::vector<int> &sizes) {
    // NOTE: this will only be called for expert batch, so we don't need to consider optional fields
    int n = sizes.size();
    std::vector<std::vector<int>> split_req_ids = split_vector_by_size(this->req_ids, sizes);
    std::vector<std::vector<int>> split_exp_ids = split_vector_by_size(this->exp_ids, sizes);
    std::vector<std::vector<int>> split_attn_dp_ranks = split_vector_by_size(this->attn_dp_ranks, sizes);
    std::vector<std::vector<int>> split_init_prefill_lens = split_vector_by_size(this->init_prefill_lens, sizes);
    std::vector<std::vector<float>> split_topk_weights = split_vector_by_size(this->topk_weights, sizes);
    std::vector<BatchMetadata> metas;
    for (int i = 0; i < n; i ++) {
        if (is_attention()) {
            // Count prefill and decode tokens for this split
            int num_prefill_tokens = std::count_if(
                split_init_prefill_lens[i].begin(), 
                split_init_prefill_lens[i].end(),
                [](int len) { return len != -1; }
            );
            int num_decode_tokens = sizes[i] - num_prefill_tokens;
            int num_prefill_seqs = num_prefill_tokens;
            metas.emplace_back(
                BatchMetadata {
                    this->batch_tag,
                    {sizes[i], this->shape[1]},
                    this->dtype, this->layer_id,
                    split_req_ids[i], 
                    split_exp_ids[i],
                    split_topk_weights[i],
                    split_attn_dp_ranks[i],
                    split_init_prefill_lens[i],
                    num_prefill_seqs,
                    num_prefill_tokens,
                    num_decode_tokens
                }
            );
        } else {
            metas.emplace_back(
                BatchMetadata {
                    this->batch_tag,
                    {sizes[i], this->shape[1]},
                    this->dtype, this->layer_id,
                    split_req_ids[i], 
                    split_exp_ids[i],
                    split_topk_weights[i],
                    split_attn_dp_ranks[i],
                    split_init_prefill_lens[i]
                }
            );
        }
    }
    return metas;
}

inline std::vector<BatchMetadata> BatchMetadata::split_by_indices(const std::vector<int> &positions) {
    // Note: will split to [positions[0], positions[1]), [positions[1], positions[2]), ..., [positions[n-1], positions[n])
    std::vector<int> sizes(positions.size() - 1);
    for (int i = 0; i < positions.size() - 1; i++) {
        sizes[i] = positions[i + 1] - positions[i];
    }
    return this->split_with_sizes(sizes);
}

inline batch_metadata_t BatchMetadata::merge_by_expert(const std::vector<batch_metadata_t> &metas, std::vector<int> &positions) {
    static std::array<int, max_num_experts> expert_cnts;

    expert_cnts.fill(0);

    std::vector<size_t> shape = metas[0]->shape;
    auto dtype = metas[0]->dtype;
    auto layer_id = metas[0]->layer_id;

    int total_tokens = 0;
    for (auto &meta: metas) {
        total_tokens += meta->num_tokens();
    }
    shape[0] = total_tokens;
    std::vector<int> req_ids(total_tokens), exp_ids(total_tokens), attn_dp_ranks(total_tokens), init_prefill_lens(total_tokens);
    std::vector<float> topk_weights(total_tokens);

    for (auto &meta: metas) {
        ASSERT (meta->num_tokens() == meta->exp_ids.size());
        for (auto &eid: meta->exp_ids) {
            expert_cnts[eid] ++;
        }
    }
    // get prefix sum of exp_cnts
    for (int i = 1; i < max_num_experts; i ++) {
        expert_cnts[i] += expert_cnts[i - 1];
    }

    int first_expert_cnt = 0;
    positions.resize(total_tokens);
    int idx = 0;
    for (auto &meta: metas) {
        // DMOE_LOG(INFO) << "merging expert metadata: " << *meta << LEND;
        for (int i = 0; i < meta->num_tokens(); i ++) {
            int pos;
            if (meta->exp_ids[i] == 0) {
                pos = first_expert_cnt ++;
            } else {
                pos = expert_cnts[meta->exp_ids[i] - 1] ++;
            }
            positions[idx] = pos; // later: tokens[pos] = tokens[idx]
            req_ids[pos] = meta->req_ids[i];
            exp_ids[pos] = meta->exp_ids[i];
            attn_dp_ranks[pos] = meta->attn_dp_ranks[i];
            init_prefill_lens[pos] = meta->init_prefill_lens[i];
            if (!meta->topk_weights.empty()) {
                topk_weights[pos] = meta->topk_weights[i];
            }
            idx ++;
        }
    }


    auto merged_meta = std::make_shared<BatchMetadata>(BatchMetadata {
        BatchTag::EXPERT, 
        shape, 
        dtype, 
        layer_id, 
        req_ids, 
        exp_ids, 
        topk_weights, 
        attn_dp_ranks, 
        init_prefill_lens 
    });

    return merged_meta;
}

inline batch_metadata_t BatchMetadata::merge_by_attention(const std::vector<batch_metadata_t> &metas) {
    int new_prefills_seqs = 0;
    int new_prefill_tokens = 0;
    int new_decode_tokens = 0;

    std::vector<int> new_req_ids{};
    std::vector<int> new_init_prefill_lens{};

    for (auto &meta: metas) {
        ASSERT (meta->attention_batch_safe_check());
        new_prefills_seqs += meta->num_prefill_seqs.value();
        new_prefill_tokens += meta->num_prefill_tokens.value();
        new_decode_tokens += meta->num_decode_tokens.value();

        for (int i = 0; i < meta->num_prefill_seqs.value(); i++) {
            new_req_ids.emplace_back(meta->req_ids[i]);
            new_init_prefill_lens.emplace_back(meta->init_prefill_lens[i]);
        }
    }

    for (auto &meta: metas) {
        for (int i = meta->num_prefill_seqs.value(); i < meta->num_prefill_seqs.value() + meta->num_decode_tokens.value(); i++) {
            new_req_ids.emplace_back(meta->req_ids[i]);
            new_init_prefill_lens.emplace_back(meta->init_prefill_lens[i]);
        }
    }
    auto new_shape = metas[0]->shape;
    new_shape[0] = new_req_ids.size();

    return std::make_shared<BatchMetadata>(BatchMetadata {
        BatchTag::ATTENTION,
        new_shape,
        metas[0]->dtype,
        metas[0]->layer_id,
        new_req_ids,
        {}, // expert_ids
        {}, // topk_weights
        {}, // attn_dp_ranks
        new_init_prefill_lens,
        new_prefills_seqs,
        new_prefill_tokens,
        new_decode_tokens
    });
}

inline batch_metadata_t BatchMetadata::pack_topk_tokens(int layer_id, const std::vector<TokenTopKInfo>& tokens) {
    int new_prefill_seqs = 0;
    int new_prefill_tokens = 0;
    int new_decode_tokens = 0;

    int n = tokens.size();

    std::vector<int> new_req_ids{};
    std::vector<int> new_init_prefill_lens{};
    std::vector<int> attn_dp_ranks{};

    for (int i = 0; i < n; i++) {
        auto &token = tokens[i];
        new_req_ids.emplace_back(token.seq_id);
        attn_dp_ranks.emplace_back(token.attn_dp_rank);
        if (token.init_prefill_len == -1) {
            new_decode_tokens ++;
            new_init_prefill_lens.emplace_back(-1);
        } else {
            new_prefill_tokens ++;
            new_prefill_seqs ++;
            new_init_prefill_lens.emplace_back(token.init_prefill_len);
        }
    }

    std::vector<size_t> new_shape{n, tokens[0].topk_tensors[0].size(-1)};

    return std::make_shared<BatchMetadata> (
        BatchMetadata {
            BatchTag::ATTENTION,
            new_shape,
            "bf16",
            layer_id,
            new_req_ids,
            {}, // exp_ids
            {}, // topk_weights
            attn_dp_ranks, // attn_dp_ranks
            new_init_prefill_lens, // init_prefill_lens
            new_prefill_seqs,
            new_prefill_tokens,
            new_decode_tokens,
        }
    );
}

#endif