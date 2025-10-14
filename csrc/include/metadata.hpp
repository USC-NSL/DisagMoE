#pragma once

#ifndef METADATA_H_
#define METADATA_H_

#include "datatypes.hpp"
#include "utils.hpp"
#include "vector_utils.hpp"

#include <vector>
#include <string>
#include <optional>
#include <memory>

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include <torch/torch.h>

constexpr int max_num_experts = 32;
constexpr int max_num_attn_dp_ranks = 32;

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
    std::vector<int> max_output_lens; // only used at attention layer 0, should be ignored later
    
    // Only used in attention batch.
    // Note: All metadata operations will ignore these optional fields.
    std::optional<int> num_prefill_tokens;
    std::optional<int> num_prefill_seqs;
    std::optional<int> num_decode_tokens;

    template<class Archive>
    void serialize(Archive &archive) {
        archive(
            batch_tag, shape, dtype, layer_id, 
            req_ids, exp_ids, topk_weights, 
            attn_dp_ranks, init_prefill_lens, max_output_lens,
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

    inline void step_layer() {
        this->layer_id ++;
    }

    BatchMetadata slice(int l, int r) {
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

    std::vector<BatchMetadata> split(const std::vector<int> &positions) {
        // Note: will split to [0, positions[0]), [positions[0], positions[1]), ..., [positions[n-1], n)
        std::vector<BatchMetadata> res;
        int start = 0;
        for (int i = 0; i < positions.size(); i++) {
            res.push_back(this->slice(start, positions[i]));
            start = positions[i];
        }
        res.push_back(this->slice(start, this->num_tokens()));
        return res;
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

    std::vector<int> get_chunk_size() const {
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
                ASSERT (0 <= i && i < req_ids.size());
                ASSERT (0 <= j && j < req_ids.size());
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
};

typedef std::shared_ptr<BatchMetadata> batch_metadata_t;

std::vector<BatchMetadata> split_with_sizes(const BatchMetadata &meta, const std::vector<int> &sizes) {
    int n = sizes.size();
    std::vector<std::vector<int>> split_req_ids = split_vector_by_size(meta.req_ids, sizes);
    std::vector<std::vector<int>> split_exp_ids = split_vector_by_size(meta.exp_ids, sizes);
    std::vector<std::vector<int>> split_attn_dp_ranks = split_vector_by_size(meta.attn_dp_ranks, sizes);
    std::vector<std::vector<int>> split_init_prefill_lens = split_vector_by_size(meta.init_prefill_lens, sizes);
    std::vector<std::vector<float>> split_topk_weights = split_vector_by_size(meta.topk_weights, sizes);
    std::vector<BatchMetadata> metas;
    for (int i = 0; i < n; i ++) {
        metas.emplace_back(
            BatchMetadata {
                meta.batch_tag,
                {sizes[i], meta.shape[1]},
                meta.dtype, meta.layer_id,
                split_req_ids[i], 
                split_exp_ids[i],
                split_topk_weights[i],
                split_attn_dp_ranks[i],
                split_init_prefill_lens[i]
            }
        );
    }
    return metas;
}

BatchMetadata merge_by_expert(const std::vector<batch_metadata_t> &metas, std::vector<int> &positions) {
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

    positions.resize(total_tokens);
    int idx = 0;
    for (auto &meta: metas) {
        // DMOE_LOG(INFO) << "merging expert metadata: " << *meta << LEND;
        for (int i = 0; i < meta->num_tokens(); i ++) {
            expert_cnts[meta->exp_ids[i]] --;
            int j = expert_cnts[meta->exp_ids[i]];
            positions[idx] = j; // later: tokens[j] = tokens[idx]
            req_ids[j] = meta->req_ids[i];
            exp_ids[j] = meta->exp_ids[i];
            attn_dp_ranks[j] = meta->attn_dp_ranks[i];
            init_prefill_lens[j] = meta->init_prefill_lens[i];
            if (!meta->topk_weights.empty()) {
                topk_weights[j] = meta->topk_weights[i];
            }
            idx ++;
        }
    }
    return BatchMetadata {
        BatchTag::EXPERT, 
        shape, 
        dtype, 
        layer_id, 
        req_ids, 
        exp_ids, 
        topk_weights, 
        attn_dp_ranks, 
        init_prefill_lens 
    };
}

BatchMetadata merge_by_attention(const std::vector<batch_metadata_t> &metas, std::vector<int> &positions) {
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
        }
    }
    auto new_shape = metas[0]->shape;
    for (int i = 1; i < metas.size(); i ++)
        new_shape[0] += metas[i]->shape[0];

    return BatchMetadata {
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
    };
}

#endif