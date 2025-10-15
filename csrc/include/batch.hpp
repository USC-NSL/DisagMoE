#pragma once

#ifndef BATCH_H_
#define BATCH_H_

#include "datatypes.hpp"
#include "metadata.hpp"
#include "vector_utils.hpp"
#include "tensor_utils.hpp"

#include <torch/torch.h>
#include <memory>

struct TokenBatch {
    torch::Tensor data;
    metadata_t metadata;

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

    static TokenBatch merge_by_expert(const std::vector<TokenBatch>& batches) {
        if (batches.empty()) {
            return TokenBatch {};
        }
        if (batches.size() == 1) {
            return batches[0];
        }

        at::cuda::CUDAStream stream = get_new_torch_stream();
        at::cuda::CUDAStreamGuard guard(stream);

        std::vector<metadata_t> metas(batches.size());
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
        
        gather_tokens_cuda(merged_tokens, srcs.data(), merged_meta->num_tokens(), merged_meta->token_hidden_dim(), stream.stream());

        return TokenBatch {merged_tokens, merged_meta};
    }

    static TokenBatch merge_by_attention(const std::vector<TokenBatch>& batches) {
        if (batches.empty()) {
            return TokenBatch {};
        }
        if (batches.size() == 1) {
            return batches[0];
        }
        AUTO_TX_RANGE;

        at::cuda::CUDAStream stream = get_new_torch_stream();
        at::cuda::CUDAStreamGuard guard(stream);

        std::vector<metadata_t> metas(batches.size());
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

        gather_tokens_cuda(merged_tokens, src_ptrs.data(), merged_meta->num_tokens(), merged_meta->token_hidden_dim(), stream.stream());

        return TokenBatch {merged_tokens, merged_meta};
    }

    static TokenBatch pack_topk_tokens(int layer_id, std::vector<TokenTopKInfo>& tokens) {
        std::sort(tokens.begin(), tokens.end(), 
            [](const TokenTopKInfo &a, const TokenTopKInfo &b) {
                if (a.init_prefill_len == -1 || b.init_prefill_len == -1) {
                    return a.init_prefill_len > b.init_prefill_len;
                }
                return a.seq_id < b.seq_id;
            }
        );

        at::cuda::CUDAStream stream = get_new_torch_stream();
        at::cuda::CUDAStreamGuard guard(stream);

        auto meta = BatchMetadata::pack_topk_tokens(layer_id, tokens);

        torch::Tensor gathered_topk_tensor = torch::empty(
            {meta->num_tokens(), meta->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        );
        // NOTE: tensor memory layout: [num_tokens, topk] flattened as 1D tensor
        std::vector<uintptr_t> src_ptrs(meta->num_tokens());

        int n = tokens.size();
        int topk = tokens[0].count();

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < topk; k++) {
                src_ptrs[k * n + i] = (uintptr_t) tokens[i].topk_tensors[k].data_ptr();
            }
        }
        gather_tokens_cuda(gathered_topk_tensor, src_ptrs.data(), meta->num_tokens(), meta->token_hidden_dim(), stream.stream());
        
        return TokenBatch{gathered_topk_tensor, meta};
    }
}

#endif