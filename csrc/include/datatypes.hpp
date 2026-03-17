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
#include "vector_utils.hpp"
#include "tensor_utils.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include <torch/torch.h>

enum class BatchTag { ATTENTION, EXPERT, TOKENIZER };

// first == layer_id, second == expert_id
using ExpertId = std::pair<int, int>;

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
};

struct ScheduleUnit { };

struct TokenMetadata: ScheduleUnit {
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

struct ParallelConfig {
    int tp = 1;
    int ep = 1;
    int dp = 1;
    int n_exp_per_rank = 1;

    // Total number of experts in the model.  When >0 the unified dispatcher
    // uses this instead of ``ep * n_exp_per_rank`` to size its routing table,
    // which is required for asymmetric expert placement where
    // ``num_experts != ep * n_exp_per_rank``.  A value of 0 means "fall back
    // to the legacy formula".
    int n_total_experts = 0;

    // (layer_id, expert_id, expert_rank)
    std::vector<std::tuple<int, int, int>> expert_ranks = {};

    ParallelConfig(int tp = 1, int ep = 1, int dp = 1, int n_exp_per_rank = 1, int n_total_experts = 0, const std::vector<std::tuple<int, int, int>> &expert_ranks = {}): 
        tp(tp), ep(ep), dp(dp), n_exp_per_rank(n_exp_per_rank), n_total_experts(n_total_experts), expert_ranks(expert_ranks) {}
};