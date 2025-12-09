#pragma once

#ifndef BATCH_H_
#define BATCH_H_

#include "datatypes.hpp"
#include "metadata.hpp"
#include "vector_utils.hpp"
#include "tensor_utils.hpp"
#include "utils.hpp"

#include <torch/torch.h>
#include <memory>

// Lazy lookup of the custom CUDA op to avoid static initialization order issues.
inline auto& get_op_gather_tokens() {
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("disag_ops::gather_tokens", "")
            .typed<void(torch::Tensor, long, long, long)>();
    return op;
}

inline auto& get_op_gather_and_sum_tokens() {
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("disag_ops::gather_and_sum_tokens", "")
            .typed<void(torch::Tensor, long, long, long, long)>();
    return op;
}

struct TokenBatch: ScheduleUnit {
    torch::Tensor data;
    batch_metadata_t metadata;

    TokenBatch() = default;

    TokenBatch(torch::Tensor data, const batch_metadata_t &metadata): data(data), metadata(metadata) {}

    std::vector<TokenBatch> split_by_expert() {
        auto chunk_sizes = metadata->get_chunk_sizes();
        auto metas = metadata->split_with_sizes(chunk_sizes);
        auto token_chunks = split_tensor_by_size(data, chunk_sizes);
        std::vector<TokenBatch> batches;
        for (int i = 0; i < chunk_sizes.size(); i ++) {
            batches.emplace_back(TokenBatch{token_chunks[i], std::make_shared<BatchMetadata>(std::move(metas[i]))});
        }
        return batches;
    }

    std::vector<TokenBatch> split_with_sizes(const std::vector<int> &sizes) {
        int n = sizes.size();
        auto metas = metadata->split_with_sizes(sizes);
        auto token_chunks = split_tensor_by_size(data, sizes);
        std::vector<TokenBatch> batches;
        for (int i = 0; i < n; i ++) {
            batches.emplace_back(TokenBatch{token_chunks[i], std::make_shared<BatchMetadata>(std::move(metas[i]))});
        }
        return batches;
    }

    inline static TokenBatch merge_by_expert(const std::vector<TokenBatch>& batches) {
        if (batches.empty()) {
            return TokenBatch {};
        }
        if (batches.size() == 1) {
            return batches[0];
        }

        std::vector<batch_metadata_t> metas(batches.size());
        for (size_t i = 0; i < batches.size(); i ++) {
            metas[i] = batches[i].metadata;
        }

        std::vector<int> mappings{};
        batch_metadata_t merged_meta = BatchMetadata::merge_by_expert(metas, mappings);

        torch::Tensor merged_tokens = torch::empty(
            {merged_meta->num_tokens(), merged_meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );

        std::vector<uintptr_t> srcs(merged_meta->num_tokens());

        int idx = 0;
        int hidden_size_bytes = merged_meta->token_hidden_dim() * merged_meta->get_datatype_size();

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

        int64_t src_ptr = reinterpret_cast<int64_t>(srcs.data());
        get_op_gather_tokens().call(merged_tokens, src_ptr, merged_meta->num_tokens(), merged_meta->token_hidden_dim());

        return TokenBatch {merged_tokens, merged_meta};
    }

    inline static TokenBatch merge_by_attention(const std::vector<TokenBatch>& batches) {
        if (batches.empty()) {
            return TokenBatch {};
        }
        if (batches.size() == 1) {
            return batches[0];
        }
        AUTO_TX_RANGE;

        std::vector<batch_metadata_t> metas(batches.size());
        for (size_t i = 0; i < batches.size(); i ++) {
            metas[i] = batches[i].metadata;
        }
        
        batch_metadata_t merged_meta = BatchMetadata::merge_by_attention(metas);

        int prefill_data_size = merged_meta->prefill_data_size();
        int decode_data_size = merged_meta->decode_data_size();

        torch::Tensor merged_tokens = torch::empty(
            {merged_meta->num_tokens(), merged_meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );

        int prefill_idx = 0;
        int decode_idx = merged_meta->num_prefill_tokens.value();

        std::vector<uintptr_t> src_ptrs(merged_meta->num_tokens());
        const int hidden_size_byte = merged_meta->token_hidden_dim() * merged_tokens.element_size();

        for (auto &batch: batches) {
            uintptr_t cur_data_ptr = (uintptr_t) batch.data.data_ptr();
            int num_prefill_tokens = batch.metadata->num_prefill_tokens.value();
            int num_decode_tokens = batch.metadata->num_decode_tokens.value();

            for (int i = 0; i < num_prefill_tokens; i ++) {
                src_ptrs[prefill_idx] = cur_data_ptr;
                cur_data_ptr += hidden_size_byte;
                prefill_idx ++;
            }

            for (int i = 0; i < num_decode_tokens; i ++) {
                src_ptrs[decode_idx] = cur_data_ptr;
                cur_data_ptr += hidden_size_byte;
                decode_idx ++;
            }
        }

        int64_t src_ptr = reinterpret_cast<int64_t>(src_ptrs.data());

        get_op_gather_tokens().call(merged_tokens, src_ptr, merged_meta->num_tokens(), merged_meta->token_hidden_dim());

        return TokenBatch {merged_tokens, merged_meta};
    }

    inline static TokenBatch merge(const std::vector<TokenBatch>& batches) {
        if (batches.empty()) {
            return TokenBatch {};
        }
        if (batches.size() == 1) {
            return batches[0];
        }

        if (batches[0].metadata->is_expert()) {
            return TokenBatch::merge_by_expert(batches);
        } else if (batches[0].metadata->is_attention()) {
            return TokenBatch::merge_by_attention(batches);
        }
        ASSERT_MSG(false, "Invalid batch metadata");
    }

    inline static TokenBatch pack_topk_tokens(int layer_id, std::vector<TokenTopKInfo>& tokens) {
        std::sort(tokens.begin(), tokens.end(), 
            [](const TokenTopKInfo &a, const TokenTopKInfo &b) {
                if (a.init_prefill_len == -1 || b.init_prefill_len == -1) {
                    return a.init_prefill_len > b.init_prefill_len;
                }
                return a.seq_id < b.seq_id;
            }
        );

        int n = tokens.size();
        int topk = tokens[0].count();

        auto meta = BatchMetadata::pack_topk_tokens(layer_id, tokens);

        ASSERT_MSG(meta->num_tokens() == n, "num_tokens is not equal to n");

        // Allocate output tensor for aggregated tokens [n, hidden_size]
        torch::Tensor aggregated_tokens_tensor = torch::empty(
            {n, meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );
        
        // NOTE: tensor memory layout: [num_tokens, topk] flattened as 1D tensor
        std::vector<uintptr_t> src_ptrs(n * topk);

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < topk; k++) {
                src_ptrs[i * topk + k] = (uintptr_t) tokens[i].topk_tensors[k].data_ptr();
            }
        }
        
        // Fused gather and sum operation
        int64_t src_ptr = reinterpret_cast<int64_t>(src_ptrs.data());
        get_op_gather_and_sum_tokens().call(aggregated_tokens_tensor, src_ptr, n, topk, meta->token_hidden_dim());
        return TokenBatch{aggregated_tokens_tensor, meta};
    }
};

#endif