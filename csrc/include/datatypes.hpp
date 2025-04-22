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
    if (a.empty()) return {};
    return std::vector<T>(a.begin() + l, a.begin() + r);
}

template<class T>
inline std::vector<std::vector<T>> split_vector(const std::vector<T> &vec, const std::vector<int> &indices) {
    std::vector<std::vector<T>> res{};
    if (vec.empty()) return {};

    for (size_t i = 0; i < indices.size() - 1; i ++) {
        int l = indices[i];
        int r = indices[i + 1];
        res.emplace_back(std::vector<T>(vec.begin() + l, vec.begin() + r));
    }
    return res;
}

inline std::vector<torch::Tensor> split_tensor(const torch::Tensor &tensor, const std::vector<int> &indices) {
    std::vector<torch::Tensor> res{};
    ASSERT (tensor.size(0) == indices.back());

    for (size_t i = 0; i < indices.size() - 1; i ++) {
        int l = indices[i];
        int r = indices[i + 1];
        res.emplace_back(tensor.slice(0, l, r));
    }
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
    int init_prefill_len;
    int topk_weight;

    template<class Archive>
    void serialize(Archive &archive) {
        archive(req_id, exp_id, attn_dp_rank, init_prefill_len);
    }

    friend std::ostream& operator<<(std::ostream &out, const TokenMetadata& token) {
        out << "TokenMetadata{req_id=" << token.req_id << ", "
            << "exp_id=" << token.exp_id << ", "
            << "attn_dp_rank=" << token.attn_dp_rank << ", "
            << "init_prefill_len=" << token.init_prefill_len << "}";
        return out;
    }
};

struct TokenTopKInfo {
    int seq_id;
    int init_prefill_len; // -1 for decoding
    int attn_dp_rank; 
    std::vector<torch::Tensor> topk_tensors;

    TokenTopKInfo() {}

    TokenTopKInfo(int seq_id, int init_prefill_len, int attn_dp_rank):
        seq_id(seq_id), init_prefill_len(init_prefill_len), attn_dp_rank(attn_dp_rank) {}

    TokenTopKInfo(int seq_id, int init_prefill_len, int attn_dp_rank, torch::Tensor tensor):
        seq_id(seq_id), init_prefill_len(init_prefill_len), 
        attn_dp_rank(attn_dp_rank),
        topk_tensors(std::vector<torch::Tensor>{tensor}) {}

    int count() const {
        return topk_tensors.size();
    }

    void append_tensor(torch::Tensor tensor) {
        topk_tensors.emplace_back(tensor);
    }

    friend std::ostream& operator<<(std::ostream &out, const TokenTopKInfo& token) {
        out << "TokenTopKInfo{seq_id=" << token.seq_id << ", "
            << "init_prefill_len=" << token.init_prefill_len << "}";
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
    std::vector<int> init_prefill_lens; // positive for first decoding tokens, -1 for subsequence decoding tokens, 0 for finished requests
    std::vector<float> topk_weights;
 
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

    inline int get_dp_rank() const {
        // NOTE: this is only used in expert worker, 
        //       caller must make sure all tokens in 
        //       the batch are from the same dp rank
        return attn_dp_ranks[0];
    }

    inline int get_expert_id() const {
        // NOTE: this is only used in expert worker,
        //       caller must make sure all tokens in
        //       the batch have the same expert id
        return exp_ids[0];
    }

    inline std::vector<int> get_expert_seg_indices() const {
        // NOTE: this is only used in expert worker, 
        //       caller must make sure all tokens are permuted by expert_id
        std::vector<int> indices{0};
        int n = exp_ids.size();
        for (int i = 1; i < n; i ++) {
            if (exp_ids[i] != exp_ids[i - 1]) {
                indices.emplace_back(i);
            }
        }
        indices.emplace_back(n);
        return indices;
    }

    constexpr size_t get_datatype_size() const {
        return 2; // bf16
    }

    inline std::vector<metadata_t> split_by_indices(const std::vector<int> &indices) {
        int n = indices.size() - 1;
        std::vector<std::vector<int>> split_req_ids = split_vector(req_ids, indices);
        std::vector<std::vector<int>> split_exp_ids = split_vector(exp_ids, indices);
        std::vector<std::vector<int>> split_attn_dp_ranks = split_vector(attn_dp_ranks, indices);
        std::vector<std::vector<int>> split_init_prefill_lens = split_vector(init_prefill_lens, indices);
        std::vector<std::vector<float>> split_topk_weights = split_vector(topk_weights, indices);
        std::vector<metadata_t> metas;
        for (int i = 0; i < n; i ++) {
            metas.emplace_back(std::make_shared<Metadata>(
                Metadata {
                    {split_req_ids[i].size(), shape[1]},
                    this->dtype, this->layer_id,
                    split_req_ids[i], split_exp_ids[i],
                    split_attn_dp_ranks[i],
                    split_init_prefill_lens[i],
                    split_topk_weights.empty() ? std::vector<float>{} : split_topk_weights[i]
                }
            ));
        }
        return metas;
    }

    inline Metadata slice(int l, int r) {
        auto shape = this->shape;
        shape[0] = r - l;

        std::vector<float> sliced_topk_weights{};
        if (topk_weights.size() > 0) {
            sliced_topk_weights = slice_vector(topk_weights, l, r);
        }

        return Metadata{
            shape, this->dtype, this->layer_id, 
            slice_vector(req_ids, l, r), 
            slice_vector(exp_ids, l, r),
            slice_vector(attn_dp_ranks, l, r),
            slice_vector(init_prefill_lens, l, r),
            sliced_topk_weights,
        };
    }

    void set_finish_signal(const std::vector<int> &continue_ids) {
        for (auto &x: init_prefill_lens) {
            x = 0;
        }
        for (auto &x: continue_ids) {
            init_prefill_lens[x] = -1;
        }
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

    inline Metadata at(const std::vector<int>& ids) const {
        auto shape = this->shape;
        shape[0] = ids.size();
        std::vector<int> req_ids_(shape[0]), exp_ids_(shape[0], -1), attn_dp_ranks_(shape[0], -1), init_prefill_lens_(shape[0], -1);
        std::vector<float> topk_weights_(shape[0]);

        for (size_t i = 0; i < ids.size(); i ++) {
            ASSERT (ids[i] < this->req_ids.size());
            req_ids_[i] = req_ids[ids[i]];
            attn_dp_ranks_[i] = attn_dp_ranks[ids[i]];
            if (exp_ids.size() >= shape[0]) {
                exp_ids_[i] = exp_ids[ids[i]];
            }
            if (init_prefill_lens.size() >= shape[0]) {
                init_prefill_lens_[i] = init_prefill_lens[ids[i]];
            }
            if (topk_weights.size() != 0) {
                topk_weights_[i] = topk_weights[ids[i]];
            }
        }
        if (topk_weights.size() == 0) {
            topk_weights_.clear();
        }
        return Metadata{
            shape, this->dtype, this->layer_id, req_ids_, exp_ids_, attn_dp_ranks_, init_prefill_lens_, topk_weights_
        };
    }

    metadata_t select_indices(const std::vector<int> &indices) {
        return std::make_shared<Metadata>(this->at(indices));
    }

    template<class Archive>
    void serialize(Archive &archive) {
        archive(shape, dtype, layer_id, req_ids, exp_ids, attn_dp_ranks, init_prefill_lens, topk_weights);
    }

    std::vector<int> get_expert_batch_sizes(int n_expert) {
        ASSERT(n_expert > 0);
        std::vector<int> batches(n_expert, 0);
        for (int eid: exp_ids)
            batches[eid] += 1;
        return batches;
    }

    void get_expert_batch_sizes_cuda(int n_expert, const std::vector<int> &inner_exp_rank, torch::Tensor tensor_cuda, uintptr_t stream_ptr) {
        AUTO_TX_RANGE;
        ASSERT(n_expert > 0);
        auto batch_sizes = get_expert_batch_sizes(n_expert);
        int64_t batches[MAX_N_EXPERTS];
        int m = inner_exp_rank.size();
        for (int i = 0; i < m; i ++) {
            ASSERT(0 <= inner_exp_rank[i] && inner_exp_rank[i] < n_expert);
            batches[i] = batch_sizes[inner_exp_rank[i]];
        }
        CUDACHECK(cudaMemcpyAsync(tensor_cuda.data_ptr(), batches, m * sizeof(int64_t), cudaMemcpyHostToDevice, cudaStream_t(stream_ptr)));
    }

    TokenMetadata info_at(int i) const {
        int exp_id = -1;
        if (exp_ids.size() > 0)
            exp_id = exp_ids[i];
        int init_prefill_len = -1;
        if (init_prefill_lens.size() > 0)
            init_prefill_len = init_prefill_lens[i];
        float topk_weight = 0;
        if (topk_weights.size() > 0)
            topk_weight = topk_weights[i];
        return TokenMetadata {req_ids[i], exp_id, attn_dp_ranks[i], init_prefill_len, topk_weight};
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
            out << "size of init_prefill_lens=" << meta.init_prefill_lens.size() << ", ";
            out << "size of topk_weights=" << meta.topk_weights.size() << ", ";
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

    // This function is only called in expert worker, so topk_weights should not be empty while using topk
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
        std::vector<int> req_ids(total_tokens), exp_ids(total_tokens), attn_dp_ranks(total_tokens), init_prefill_lens(total_tokens);
        std::vector<float> topk_weights(total_tokens);

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
            // DMOE_LOG(INFO) << "merging expert metadata: " << *meta << LEND;
            for (int i = 0; i < meta->num_tokens(); i ++) {
                exp_cnts[meta->exp_ids[i]] --;
                int j = exp_cnts[meta->exp_ids[i]];
                mappings[idx] = j; // tokens[i] = tokens[mapping[i]]
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
        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids, exp_ids, attn_dp_ranks, init_prefill_lens, topk_weights
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

    void shrink_topk(int topk) {
        if (topk == 1)
            return;
        this->shape[0] /= topk;
    }

    void duplicate_topk(int topk) {
        if (topk == 1)
            return;
        std::vector<int> new_req_ids, new_init_prefill_lens, new_attn_dp_ranks;
        int n = req_ids.size();
        new_req_ids.reserve(n * topk);
        new_init_prefill_lens.reserve(n * topk);
        new_attn_dp_ranks.reserve(n * topk);

        for (int j = 0; j < topk; j++) {
            for (int i = 0; i < n; i++) {
                new_req_ids.push_back(req_ids[i]);
                new_init_prefill_lens.push_back(init_prefill_lens[i]);
                new_attn_dp_ranks.push_back(attn_dp_ranks[i]);
            }
        }
        req_ids = std::move(new_req_ids);
        init_prefill_lens = std::move(new_init_prefill_lens);
        attn_dp_ranks = std::move(new_attn_dp_ranks);
        shape[0] *= topk;
    }

    void permute_token_infos(const std::vector<int> &exp_mappings) {
        if (exp_mappings.empty())
            return;
        ASSERT (req_ids.size() == exp_mappings.size());
        std::vector<int> tmp_int(exp_mappings.size());
        #define MOVE_INT(a) { \
            for (int i = 0; i < exp_mappings.size(); i ++) {    \
                int j = exp_mappings[i];                        \
                ASSERT(0 <= j && j < tmp_int.size());               \
                tmp_int[j] = a[i];                                  \
            }                                                   \
            a = tmp_int;                                            \
        }

        std::vector<float> tmp_float(exp_mappings.size());
        #define MOVE_FLOAT(a) { \
            for (int i = 0; i < exp_mappings.size(); i ++) {    \
                int j = exp_mappings[i];                        \
                ASSERT(0 <= j && j < tmp_float.size());               \
                tmp_float[j] = a[i];                                  \
            }                                                   \
            a = tmp_float;                                            \
        }
        MOVE_INT(exp_ids);
        MOVE_INT(req_ids);
        MOVE_INT(attn_dp_ranks);
        MOVE_INT(init_prefill_lens);
        if (!topk_weights.empty()) {
            MOVE_FLOAT(topk_weights);
        }
        #undef MOVE_INT
        #undef MOVE_FLOAT
    }

    std::vector<int> sort_by_prefill_order() {
        std::vector<int> rank(req_ids.size()), mapping(req_ids.size());
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
                    slice_vector(init_prefill_lens, 0, p),
                    topk_weights.empty() ? std::vector<float>{} : slice_vector(topk_weights, 0, p)
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
                    slice_vector(init_prefill_lens, p, -1),
                    topk_weights.empty() ? std::vector<float>{} : slice_vector(topk_weights, p, -1)
                }
            )
        );
    }

    static metadata_t pack_tokens(int layer_id, const std::vector<TokenTopKInfo>& tokens) {
        int topk = tokens[0].count();
        int n = tokens.size();

        std::vector<int> req_ids, exp_ids, attn_dp_ranks, init_prefill_lens;
        std::vector<size_t> shape = {n * topk, tokens[0].topk_tensors[0].size(0)};
        std::string dtype = "bf16";

        for (int i = 0; i < n; i++) {
            auto &token = tokens[i];
            req_ids.push_back(token.seq_id);
            exp_ids.push_back(token.init_prefill_len);
            attn_dp_ranks.push_back(token.attn_dp_rank);
            init_prefill_lens.push_back(token.init_prefill_len);
        }

        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids, exp_ids, attn_dp_ranks, init_prefill_lens, {}
        });
    }

    std::vector<TokenTopKInfo> unpack_tokens() {
        std::vector<TokenTopKInfo> tokens;
        tokens.reserve(req_ids.size());
        for (int i = 0; i < req_ids.size(); i ++) {
            tokens.emplace_back(req_ids[i], init_prefill_lens[i], attn_dp_ranks[i]);
        }
        return tokens;
    }
};

struct TensorBatch {
    torch::Tensor data;
    metadata_t metadata;

    static std::vector<TensorBatch> split_by_expert_id(torch::Tensor tensor, metadata_t meta) {
        auto seg_indices = meta->get_expert_seg_indices();
        int n = seg_indices.size() - 1;
        if (n == 1) {
            // only one segment
            return {TensorBatch{tensor, meta}};
        }
        std::vector<TensorBatch> batches;
        auto metas = meta->split_by_indices(seg_indices);
        auto tensors = split_tensor(tensor, seg_indices);
        for (int i = 0; i < n; i++) {
            batches.emplace_back(TensorBatch{tensors[i], metas[i]});
        }
        return batches;
    }

    std::vector<TensorBatch> split_by_expert_id() {
        return TensorBatch::split_by_expert_id(this->data, this->metadata);
    }

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

    static TensorBatch pack_tokens(int layer_id, const std::vector<TokenTopKInfo>& tokens) {
        metadata_t meta = Metadata::pack_tokens(layer_id, tokens);

        torch::Tensor tensor = torch::empty(
            {meta->num_tokens(), meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );

        std::vector<uintptr_t> srcs(meta->num_tokens());

        at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(true, -1);
        cudaStream_t stream = c10_stream.stream();
        at::cuda::CUDAStreamGuard guard(c10_stream);

        int idx = 0;
        int hidden_size_bytes = meta->token_hidden_dim() * meta->get_datatype_size();

        {
            tx_range _{"TensorBatch::pack_tokens::perpare_for_gather_cuda"};
            for (auto &token: tokens) {
                uintptr_t cur_ptr = (uintptr_t) token.topk_tensors[0].data_ptr();
                for (int i = 0; i < token.topk_tensors.size(); i ++) {
                    srcs[idx] = cur_ptr;
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

    std::vector<int> init_prefill_lens; // per perfill seq, length of (num_prefill_seqs)

    std::vector<uint8_t> expert_ids; // optional, per token, length of (num_prefill_tokens + num_decode_tokens)

    std::vector<float> topk_weights; // optional, length of (num_prefill_tokens + num_decode_tokens) * topk

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

        ASSERT(num_prefill_tokens == num_prefill_seqs);
        ASSERT(p > 0);
        if (p < num_prefill_tokens) {
            ASSERT(seq_ids.size() >= p);
            ASSERT(init_prefill_lens.size() >= p);
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
                        slice_vector(init_prefill_lens, 0, p),
                        !expert_ids.empty() ? slice_vector(expert_ids, 0, p) : std::vector<uint8_t>{},
                        !topk_weights.empty() ? slice_vector(topk_weights, 0, p) : std::vector<float>{},
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
                        slice_vector(init_prefill_lens, p, -1),
                        !expert_ids.empty() ? slice_vector(expert_ids, p, -1) : std::vector<uint8_t>{},
                        !topk_weights.empty() ? slice_vector(topk_weights, p, -1) : std::vector<float>{},
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
                        init_prefill_lens,
                        !expert_ids.empty() ? slice_vector(expert_ids, 0, p) : std::vector<uint8_t>{},
                        !topk_weights.empty() ? slice_vector(topk_weights, 0, p) : std::vector<float>{},
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
                        !expert_ids.empty() ? slice_vector(expert_ids, p, -1) : std::vector<uint8_t>{},
                        !topk_weights.empty() ? slice_vector(topk_weights, p, -1) : std::vector<float>{},
                        slice_vector(attn_dp_ranks, p, -1)
                    }
                )
            );
        }
    }

    void shrink_topk(int topk) {
        if (topk == 1)
            return;
        this->shape[0] /= topk;
    }

    static attn_metadata_t merge(const std::vector<attn_metadata_t>& batches) {
        AUTO_TX_RANGE;
        
        int new_prefills_seqs = 0;
        int new_prefill_tokens = 0;
        int new_decode_tokens = 0;

        std::vector<int> new_seq_ids{};
        std::vector<uint8_t> new_attn_dp_ranks{};
        std::vector<int> new_init_prefill_lens{};

        for (auto &batch: batches) {
            new_prefills_seqs += batch->num_prefill_seqs;
            new_prefill_tokens += batch->num_prefill_tokens;
            new_decode_tokens += batch->num_decode_tokens;

            for (int i = 0; i < batch->num_prefill_seqs; i++) {
                new_seq_ids.emplace_back(batch->seq_ids[i]);
                new_attn_dp_ranks.emplace_back(batch->attn_dp_ranks[i]);
                new_init_prefill_lens.emplace_back(batch->init_prefill_lens[i]);
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
                new_init_prefill_lens,
                {}, // expert_ids
                {}, // topk_weights
                new_attn_dp_ranks,
            }
        );
    }

    static attn_metadata_t pack_tokens(int layer_id, const std::vector<TokenTopKInfo>& tokens) {
        int new_prefill_seqs = 0;
        int new_prefill_tokens = 0;
        int new_decode_tokens = 0;

        int topk = tokens[0].count();
        int n = tokens.size();

        std::vector<int> new_seq_ids{};
        std::vector<int> new_init_prefill_lens{};
        std::vector<uint8_t> attn_dp_ranks{};

        for (int i = 0; i < n; i++) {
            auto &token = tokens[i];
            new_seq_ids.emplace_back(token.seq_id);
            attn_dp_ranks.emplace_back(token.attn_dp_rank);
            if (token.init_prefill_len == -1) {
                new_decode_tokens ++;
            } else {
                // NOTE: Only considered for prefill length = 1
                new_prefill_tokens ++;
                new_prefill_seqs ++;
                new_init_prefill_lens.emplace_back(token.init_prefill_len);
            }
        }

        std::vector<size_t> new_shape{n * topk, tokens[0].topk_tensors[0].size(-1)};

        return std::make_shared<AttentionBatchMetadata> (
            AttentionBatchMetadata {
                layer_id,
                new_shape,
                "bf16",
                new_prefill_seqs,
                new_prefill_tokens,
                new_decode_tokens,
                new_seq_ids,
                new_init_prefill_lens,
                {}, // expert_ids
                {}, // topk_weights
                attn_dp_ranks // attn_dp_ranks
            }
        );
    }

    metadata_t to_metadata() {
        auto shape = this->shape;
        auto dtype = this->dtype;
        auto layer_id = this->layer_id;
        std::vector<int> req_ids_;
        std::vector<int> attn_dp_ranks_;
        std::vector<int> init_prefill_lens_;

        // DMOE_LOG(INFO) << "To metadata, seq_ids: ";
        // for (int i = 0; i < num_prefill_seqs + num_decode_tokens; i ++)
        //     std::cout << seq_ids[i] << " ";
        // std::cout << LEND;
        
        for (int i = 0; i < num_prefill_seqs; i ++) {
            req_ids_.push_back(seq_ids[i]);
            attn_dp_ranks_.push_back(attn_dp_ranks[i]);
            init_prefill_lens_.push_back(init_prefill_lens[i]);
        }

        for (int i = 0; i < num_decode_tokens; i ++) {
            req_ids_.push_back(seq_ids[num_prefill_seqs + i]);
            attn_dp_ranks_.push_back(attn_dp_ranks[num_prefill_seqs + i]);
            init_prefill_lens_.push_back(-1);
        }
        
        return std::make_shared<Metadata>(Metadata {
            shape, dtype, layer_id, req_ids_, {}, attn_dp_ranks_, init_prefill_lens_
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
        if (batches.size() == 1) {
            return batches[0];
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

    static AttentionBatch pack_tokens(int layer_id, std::vector<TokenTopKInfo>& tokens) {

        // std::cerr << "AttentionBatch::pack_tokens, tokens before sorting: ";
        // for (auto &token: tokens) {
        //     std::cerr << token << " ";
        // }
        // std::cerr << std::endl;
        std::sort(tokens.begin(), tokens.end(), 
            [](const TokenTopKInfo &a, const TokenTopKInfo &b) {
                if (a.init_prefill_len == -1 || b.init_prefill_len == -1) {
                    return a.init_prefill_len > b.init_prefill_len;
                }
                return a.seq_id < b.seq_id;
            }
        );

        // std::cerr << "AttentionBatch::pack_tokens, tokens after sorting: ";
        // for (auto &token: tokens) {
        //     std::cerr << token << " ";
        // }
        // std::cerr << std::endl;

        // DMOE_LOG(INFO) << "AttentionBatch::pack_tokens after sorting, layer_id=" << layer_id << ", tokens " << tokens[0].seq_id << LEND;

        at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(true, -1);
        cudaStream_t stream = c10_stream.stream();
        at::cuda::CUDAStreamGuard guard(c10_stream);

        auto meta = AttentionBatchMetadata::pack_tokens(layer_id, tokens);

        // DMOE_LOG(INFO) << "tokens packed in to attn batch, meta=" << meta->seq_ids[0] << LEND;

        torch::Tensor gathered_topk_tensor = torch::empty(
            {meta->num_tokens(), meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );
        // tensor memory layout: [top1, ..., top1, top2, ..., top2, ..., topk, ..., topk]
        std::vector<uintptr_t> src_ptrs(meta->num_tokens());

        int n = tokens.size();
        int topk = tokens[0].count();

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < topk; k++) {
                src_ptrs[k * n + i] = (uintptr_t) tokens[i].topk_tensors[k].data_ptr();
            }
        }
        gather_tokens_cuda(gathered_topk_tensor, src_ptrs.data(), meta->num_tokens(), meta->token_hidden_dim(), stream);
        
        return AttentionBatch{gathered_topk_tensor, meta};
    }

};

struct SloStat {
    int req_id;
    clock_t t_prefill;  // time to all finished prefill tokens, in us
    clock_t t_prefill_std; // the same as t_prefill, but use chrono instead of clock(), in miliseconds
    clock_t t_decode;   // time to all finished decode tokens, in us

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