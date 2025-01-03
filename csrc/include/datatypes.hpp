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
    std::vector<T> res{};
    res.clear();
    if (r < 0)
        r = a.size();
    if (l == r)
        return {};
    if (a.empty())
        return {};
    ASSERT(l <= r);
    ASSERT(r <= a.size());
    res.reserve(r - l);
    for (auto i = a.begin() + l; i != a.begin() + r; i ++)
        res.emplace_back(*i);
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
    int attn_dp_rank;

    ChannelInfo() {}
    ChannelInfo(const std::vector<ExpertId> &expert_ids,
                const std::vector<int> &attn_layer_ids,
                int attn_dp_rank):
                expert_ids(expert_ids), attn_layer_ids(attn_layer_ids), attn_dp_rank(attn_dp_rank)
    {}

    inline bool is_sampler_channel() {
        return expert_ids.empty() && attn_layer_ids.empty();
    }
};

struct TokenMetadata {
    int req_id;
    int exp_id;
    int attn_dp_rank;
    int prefill_pos;

    template<class Archive>
    void serialize(Archive &archive) {
        archive(req_id, exp_id, attn_dp_rank, prefill_pos);
    }

    friend std::ostream& operator<<(std::ostream &out, const TokenMetadata& token) {
        out << "TokenMetadata{req_id=" << token.req_id << ", "
            << "exp_id=" << token.exp_id << ", "
            << "attn_dp_rank=" << token.attn_dp_rank << ", "
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
    std::vector<int> attn_dp_ranks;
    std::vector<int> prefill_poss;
 
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
            slice_vector(attn_dp_ranks, l, r),
            slice_vector(prefill_poss, l, r),
        };
    }

    inline Metadata at(const std::vector<int>& ids) const {
        auto shape = this->shape;
        shape[0] = ids.size();
        std::vector<int> req_ids_(shape[0]), exp_ids_(shape[0], -1), attn_dp_ranks_(shape[0], -1), prefill_poss_(shape[0], -1);

        for (size_t i = 0; i < ids.size(); i ++) {
            ASSERT (ids[i] < this->req_ids.size());
            req_ids_[i] = req_ids[ids[i]];
            attn_dp_ranks_[i] = attn_dp_ranks[ids[i]];
            if (exp_ids.size() >= shape[0]) {
                exp_ids_[i] = exp_ids[ids[i]];
            }
            if (prefill_poss.size() >= shape[0]) {
                prefill_poss_[i] = prefill_poss[ids[i]];
            }
        }
        return Metadata{
            shape, this->dtype, this->layer_id, req_ids_, exp_ids_, attn_dp_ranks_, prefill_poss_
        };
    }

    template<class Archive>
    void serialize(Archive &archive) {
        archive(shape, dtype, layer_id, req_ids, exp_ids, attn_dp_ranks, prefill_poss);
    }

    std::vector<int> get_expert_batch_sizes(int n_expert) {
        ASSERT(n_expert > 0);
        std::vector<int> batches(n_expert, 0);
        for (int eid: exp_ids)
            batches[eid] += 1;
        return batches;
    }

    void get_expert_batch_sizes_cuda(int n_expert, const std::vector<int> &inner_exp_rank, torch::Tensor tensor_cuda) {
        AUTO_TX_RANGE;
        ASSERT(n_expert > 0);
        auto batch_sizes = get_expert_batch_sizes(n_expert);
        int64_t batches[MAX_N_EXPERTS];
        int m = inner_exp_rank.size();
        for (int i = 0; i < m; i ++) {
            ASSERT(0 <= inner_exp_rank[i] && inner_exp_rank[i] < n_expert);
            batches[i] = batch_sizes[inner_exp_rank[i]];
        }
        CUDACHECK(cudaMemcpyAsync(tensor_cuda.data_ptr(), batches, m * sizeof(int64_t), cudaMemcpyHostToDevice));
    }

    TokenMetadata info_at(int i) const {
        int exp_id = -1;
        if (exp_ids.size() > 0)
            exp_id = exp_ids[i];
        int prefill_pos = -1;
        if (prefill_poss.size() > 0)
            prefill_pos = prefill_poss[i];
        return TokenMetadata {req_ids[i], exp_id, attn_dp_ranks[i], prefill_pos};
    }

    friend std::ostream& operator<<(std::ostream &out, const Metadata& meta) {
        out << "Metadata{";
        {
            out << "shape=(" << meta.shape[0];
            for (size_t i = 1; i < meta.shape.size(); i ++)
                out << ", " << meta.shape[i];
            out << "), ";

            out << "layer_id=" << meta.layer_id << ", ";
            out << "num_reqs=" << meta.req_ids.size() << ", ";
            out << "size of exp_ids=" << meta.exp_ids.size() << ", ";
            out << "size of prefill_poss=" << meta.prefill_poss.size() << ", ";
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

    static metadata_t merge_by_exp_ids(const std::vector<metadata_t> &metas, std::vector<int> &mappings) {
        AUTO_TX_RANGE;

        constexpr int MAX_NUM_EXPERTS = 32;
        std::vector<size_t> shape = metas[0]->shape;
        auto dtype = metas[0]->dtype;
        auto layer_id = metas[0]->layer_id;

        int total_tokens = 0;
        for (auto &meta: metas) {
            total_tokens += meta->num_tokens();
        }
        shape[0] = total_tokens;
        std::vector<int> req_ids(total_tokens), exp_ids(total_tokens), attn_dp_ranks(total_tokens), prefill_poss(total_tokens);

        std::vector<int> exp_cnts(MAX_NUM_EXPERTS, 0);
        for (auto &meta: metas) {
            ASSERT (meta->num_tokens() == meta->exp_ids.size());
            for (auto &eid: meta->exp_ids) {
                exp_cnts[eid] ++;
            }
        }
        // get prefix sum of exp_cnts
        for (int i = 1; i < MAX_NUM_EXPERTS; i ++) {
            exp_cnts[i] += exp_cnts[i - 1];
        }

        mappings.resize(total_tokens);
        int idx = 0;
        for (auto &meta: metas) {
            for (int i = 0; i < meta->num_tokens(); i ++) {
                exp_cnts[meta->exp_ids[i]] --;
                int j = exp_cnts[meta->exp_ids[i]];
                mappings[idx] = j; // tokens[i] = tokens[mapping[i]]
                req_ids[j] = meta->req_ids[i];
                exp_ids[j] = meta->exp_ids[i];
                attn_dp_ranks[j] = meta->attn_dp_ranks[i];
                prefill_poss[j] = meta->prefill_poss[i];
                idx ++;
            }
        }
        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids, exp_ids, attn_dp_ranks, prefill_poss
        });
    }

    // NOTE: if exp_ids exists, merge by exp_ids; else merge by default order
    static metadata_t merge(const std::vector<metadata_t> &metas) {
        AUTO_TX_RANGE;

        ASSERT(metas.size() > 0);
        std::vector<size_t> shape = metas[0]->shape;
        auto dtype = metas[0]->dtype;
        auto layer_id = metas[0]->layer_id;

        int total_tokens = 0;
        for (auto &meta: metas) {
            total_tokens += meta->num_tokens();
        }
        shape[0] = total_tokens;
        std::vector<int> req_ids, exp_ids, attn_dp_ranks, prefill_poss;
        
        req_ids.reserve(total_tokens);
        exp_ids.reserve(total_tokens);
        attn_dp_ranks.reserve(total_tokens);
        prefill_poss.reserve(total_tokens);

        for (size_t i = 0; i < metas.size(); i ++) {
            auto meta = metas[i];
            extend(req_ids, meta->req_ids);
            extend(exp_ids, meta->exp_ids);
            extend(attn_dp_ranks, meta->attn_dp_ranks);
            extend(prefill_poss, meta->prefill_poss);
        }

        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids, exp_ids, attn_dp_ranks, prefill_poss
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
        MOVE(attn_dp_ranks);
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
                return (attn_dp_ranks[i] != attn_dp_ranks[j]) ? attn_dp_ranks[i] < attn_dp_ranks[j] : 
                    (prefill_poss[i] < 0 || prefill_poss[j] < 0) ?
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

    std::pair<metadata_t, metadata_t> split(int p) {
        ASSERT(p > 0);
        ASSERT(p < shape[0]);
        return std::make_pair(
            std::make_shared<Metadata> (
                Metadata {
                    {(size_t) p, shape[1]},
                    dtype,
                    layer_id,
                    slice_vector(req_ids, 0, p),
                    slice_vector(exp_ids, 0, p),
                    slice_vector(attn_dp_ranks, 0, p),
                    slice_vector(prefill_poss, 0, p)
                }
            ),
            std::make_shared<Metadata> (
                Metadata {
                    {(size_t) (shape[0] - p), shape[1]},
                    dtype,
                    layer_id,
                    slice_vector(req_ids, p, -1),
                    slice_vector(exp_ids, p, -1),
                    slice_vector(attn_dp_ranks, p, -1),
                    slice_vector(prefill_poss, p, -1)
                }
            )
        );
    }
};

struct TensorBatch {
    torch::Tensor data;
    metadata_t metadata;

    // NOTE: merge by exp_ids, this function is only called in expert worker
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

        std::vector<int> mappings{};
        auto meta = Metadata::merge_by_exp_ids(metas, mappings);

        torch::Tensor tensor = torch::empty(
            {meta->num_tokens(), meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );

        std::vector<uintptr_t> srcs(meta->num_tokens());

        int idx = 0;
        int hidden_size_bytes = meta->token_hidden_dim() * meta->get_datatype_size();

        {
            tx_range _{"TensorBatch::merge::perpare_for_gather_cuda"};
            for (auto &batch: batches) {
                uintptr_t cur_ptr = (uintptr_t) batch.data.data_ptr();
                for (int i = 0; i < batch.metadata->num_tokens(); i ++) {
                    srcs[mappings[idx]] = cur_ptr;
                    cur_ptr += hidden_size_bytes;
                    idx ++;
                }
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

    std::vector<uint8_t> attn_dp_ranks; // per token, length of (num_prefill_seqs + num_decode_tokens)

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

    std::pair<attn_metadata_t, attn_metadata_t> split(int p) {
        /*
            Split the Metadata into [0, p) and [p, n)
        */

        // TODO(hogura|20241206): here we assume #prefill_len=1
        ASSERT(num_prefill_tokens == num_prefill_seqs);
        ASSERT(p > 0);
        if (p < num_prefill_tokens) {
            ASSERT(seq_ids.size() >= p);
            ASSERT(prefill_seq_len.size() >= p);
            ASSERT(prefill_query_len.size() >= p);
            return std::make_pair(
                std::make_shared<AttentionBatchMetadata> (
                    AttentionBatchMetadata {
                        layer_id,
                        {(size_t) p, shape[1]},
                        dtype,
                        p,
                        p,
                        0,
                        slice_vector(seq_ids, 0, p),
                        slice_vector(prefill_seq_len, 0, p),
                        slice_vector(prefill_query_len, 0, p),
                        !expert_ids.empty() ? slice_vector(expert_ids, 0, p) : std::vector<uint8_t>{},
                        slice_vector(attn_dp_ranks, 0, p)
                    }
                ),
                std::make_shared<AttentionBatchMetadata> (
                    AttentionBatchMetadata {
                        layer_id,
                        {(size_t) (shape[0] - p), shape[1]},
                        dtype,
                        num_prefill_seqs - p,
                        num_prefill_tokens - p,
                        num_decode_tokens,
                        slice_vector(seq_ids, p, -1),
                        slice_vector(prefill_seq_len, p, -1),
                        slice_vector(prefill_query_len, p, -1),
                        !expert_ids.empty() ? slice_vector(expert_ids, p, -1) : std::vector<uint8_t>{},
                        slice_vector(attn_dp_ranks, p, -1)
                    }
                )
            );
        } else {
            return std::make_pair(
                std::make_shared<AttentionBatchMetadata> (
                    AttentionBatchMetadata {
                        layer_id,
                        {(size_t) p, shape[1]},
                        dtype,
                        num_prefill_seqs,
                        num_prefill_tokens,
                        p - num_prefill_tokens,
                        slice_vector(seq_ids, 0, p),
                        prefill_seq_len,
                        prefill_query_len,
                        !expert_ids.empty() ? slice_vector(expert_ids, 0, p) : std::vector<uint8_t>{},
                        slice_vector(attn_dp_ranks, 0, p),
                    }
                ),
                std::make_shared<AttentionBatchMetadata> (
                    AttentionBatchMetadata {
                        layer_id,
                        {(size_t) (shape[0] - p), shape[1]},
                        dtype,
                        0,
                        0,
                        num_decode_tokens - (p - num_prefill_tokens),
                        slice_vector(seq_ids, p, -1),
                        std::vector<int>{},
                        std::vector<int>{},
                        !expert_ids.empty() ? slice_vector(expert_ids, p, -1) : std::vector<uint8_t>{},
                        slice_vector(attn_dp_ranks, p, -1)
                    }
                )
            );
        }
    }

    static attn_metadata_t merge(const std::vector<attn_metadata_t>& batches) {
        AUTO_TX_RANGE;
        
        int new_prefills_seqs = 0;
        int new_prefill_tokens = 0;
        int new_decode_tokens = 0;

        std::vector<int> new_seq_ids{};
        std::vector<uint8_t> new_attn_dp_ranks{};
        std::vector<int> new_prefill_seq_len{};
        std::vector<int> new_prefill_query_len{};

        for (auto &batch: batches) {
            new_prefills_seqs += batch->num_prefill_seqs;
            new_prefill_tokens += batch->num_prefill_tokens;
            new_decode_tokens += batch->num_decode_tokens;

            for (int i = 0; i < batch->num_prefill_seqs; i++) {
                new_seq_ids.emplace_back(batch->seq_ids[i]);
                new_attn_dp_ranks.emplace_back(batch->attn_dp_ranks[i]);
                new_prefill_seq_len.emplace_back(batch->prefill_seq_len[i]);
                new_prefill_query_len.emplace_back(batch->prefill_query_len[i]);
            }
        }

        for (auto &batch: batches) {
            for (int i = batch->num_prefill_seqs; i < batch->num_prefill_seqs + batch->num_decode_tokens; i++) {
                new_seq_ids.emplace_back(batch->seq_ids[i]);
                new_attn_dp_ranks.emplace_back(batch->attn_dp_ranks[i]);
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
                new_prefill_query_len,
                {}, // expert_ids
                new_attn_dp_ranks,
            }
        );
    }

    metadata_t to_metadata() {
        auto shape = this->shape;
        auto dtype = this->dtype;
        auto layer_id = this->layer_id;
        std::vector<int> req_ids_;
        std::vector<int> attn_dp_ranks_;
        std::vector<int> prefill_poss_;

        // DMOE_LOG(INFO) << "To metadata, seq_ids: ";
        // for (int i = 0; i < num_prefill_seqs + num_decode_tokens; i ++)
        //     std::cout << seq_ids[i] << " ";
        // std::cout << LEND;
        
        for (int i = 0; i < num_prefill_seqs; i ++) {
            // TODO(hogura|20241014): modify to chunked prefill
            for (int j = 0; j < prefill_seq_len[i]; j ++) {
                req_ids_.push_back(seq_ids[i]);
                attn_dp_ranks_.push_back(attn_dp_ranks[i]);
                prefill_poss_.push_back(j);
                // TODO(hogura|20241014): add attention replica
            }
        }

        for (int i = 0; i < num_decode_tokens; i ++) {
            req_ids_.push_back(seq_ids[num_prefill_seqs + i]);
            attn_dp_ranks_.push_back(attn_dp_ranks[num_prefill_seqs + i]);
            prefill_poss_.push_back(-1);
        }
        
        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids_, {}, attn_dp_ranks_, prefill_poss_
        });
    }
};


struct AttentionBatch {
    torch::Tensor data;
    attn_metadata_t metadata;

    static AttentionBatch merge(const std::vector<AttentionBatch>& batches) {
        if (batches.empty()) {
            return AttentionBatch {};
        }
        at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(true, -1);
        cudaStream_t stream = c10_stream.stream();
        at::cuda::CUDAStreamGuard guard(c10_stream);
        AUTO_TX_RANGE;

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

        int prefill_idx = 0;
        int decode_idx = meta->num_prefill_tokens;

        std::vector<uintptr_t> src_ptrs(meta->num_tokens());
        const int hidden_size_byte = meta->token_hidden_dim() * tensor.element_size();

        {
            tx_range _{"AttentionBatch::merge::perpare_for_gather_cuda"};
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
        }
        gather_tokens_cuda(tensor, src_ptrs.data(), meta->num_tokens(), meta->token_hidden_dim(), stream);

        return AttentionBatch {tensor, meta};
    }
};

struct SloStat {
    int req_id;
    clock_t t_prefill;  // time to all finished prefill tokens
    clock_t t_decode;   // time to all finished decode tokens

    std::vector<clock_t> t_tokens;
};

struct ParallelConfig {
    int tp = 1;
    int ep = 1;
    int dp = 1;
    int n_exp_per_rank = 1;

    // (layer_id, expert_id, expert_rank)
    std::vector<std::tuple<int, int, int>> expert_ranks = {};

    ParallelConfig(int tp = 1, int ep = 1, int dp = 1, int n_exp_per_rank = 1, const std::vector<std::tuple<int, int, int>> &expert_ranks = {}): 
        tp(tp), ep(ep), dp(dp), n_exp_per_rank(n_exp_per_rank), expert_ranks(expert_ranks) {}
};