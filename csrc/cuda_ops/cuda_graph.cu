#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

using namespace at;

using bfloat16_t = __nv_bfloat16;

template<int TOKENS_PER_BLOCK>
__global__ void preprocess_fused_cuda(
    const bfloat16_t* __restrict__ hidden,
    const int* __restrict__ block_tables,
    const long* __restrict__ positions,
    const long* __restrict__ slot_mapping,
    const int* __restrict__ seq_lens,
    const int* __restrict__ context_lens,
    const int* __restrict__ seq_start_loc,
    int in_block_table_stride,

    bfloat16_t* __restrict__ out_hidden,
    int* __restrict__ out_block_tables,
    long* __restrict__ out_positions,
    long* __restrict__ out_slot_mapping,
    int* __restrict__ out_seq_lens,
    int* __restrict__ out_context_lens,
    int* __restrict__ out_seq_start_loc,
    int out_block_table_stride,

    int T, int H, int B, int padded_batch_size
){
    int num_blocks = gridDim.x;
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    if (block_id == num_blocks - 1) {
        // Last block deals with metadata tensors
        #pragma unroll
        for (int i = thread_id; i < padded_batch_size; i += num_threads) {
            if (i < T) {
                out_positions[i]    = positions[i];
                out_slot_mapping[i] = slot_mapping[i];
                out_seq_lens[i]     = seq_lens[i];
                out_context_lens[i] = context_lens[i];
            } else {
                out_positions[i]    = 0;
                out_slot_mapping[i] = -1;
                out_seq_lens[i]     = 0;
                out_context_lens[i] = 0;
            }
        }

        #pragma unroll
        for (int i = thread_id; i < padded_batch_size + 1; i += num_threads) {
            if (i < T+1) {
                out_seq_start_loc[i] = seq_start_loc[i];
            } else {
                out_seq_start_loc[i] = 0;
            }
        }
    } else {
        // Other blocks deal with data tensors and block tables
        using hidden_vec_t = float4;
        constexpr int VEC_SIZE_HIDDEN = sizeof(hidden_vec_t) / sizeof(bfloat16_t);

        using block_table_vec_t = int;
        constexpr int VEC_SIZE_BLOCK_TABLE = sizeof(block_table_vec_t) / sizeof(int);

        int start_token = block_id * TOKENS_PER_BLOCK;

        for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
            int token = start_token + t;
            if (token >= T) return;

            const bfloat16_t* src_hidden = hidden + token * H;
            bfloat16_t* dst_hidden       = out_hidden + token * H;

            #pragma unroll
            for (int i = thread_id * VEC_SIZE_HIDDEN; i < H; i += num_threads * VEC_SIZE_HIDDEN) {
                int offset = i / VEC_SIZE_HIDDEN;
                hidden_vec_t v = reinterpret_cast<const hidden_vec_t*>(src_hidden)[offset];
                reinterpret_cast<hidden_vec_t*>(dst_hidden)[offset] = v;
            }

            const int* src_bt = block_tables + token * in_block_table_stride;
            int* dst_bt       = out_block_tables + token * out_block_table_stride;

            int vec_end = (B / VEC_SIZE_BLOCK_TABLE) * VEC_SIZE_BLOCK_TABLE; 

            #pragma unroll
            for (int i = thread_id * VEC_SIZE_BLOCK_TABLE; i < vec_end; i += num_threads * VEC_SIZE_BLOCK_TABLE) {
                int offset = i / VEC_SIZE_BLOCK_TABLE;
                block_table_vec_t v = reinterpret_cast<const block_table_vec_t*>(src_bt)[offset];
                reinterpret_cast<block_table_vec_t*>(dst_bt)[offset] = v;
            }

            for (int i = vec_end + thread_id; i < B; i += num_threads) {
                dst_bt[i] = src_bt[i];
            }
        }
    }
}

template<int TOKENS_PER_BLOCK>
void launch_preprocess_fused_cuda(
    const at::Tensor& hidden,
    const at::Tensor& block_tables,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& seq_lens,
    const at::Tensor& context_lens,
    const at::Tensor& seq_start_loc,

    at::Tensor& out_hidden,
    at::Tensor& out_block_tables,
    at::Tensor& out_positions,
    at::Tensor& out_slot_mapping,
    at::Tensor& out_seq_lens,
    at::Tensor& out_context_lens,
    at::Tensor& out_seq_start_loc,
    int padded_batch_size
){
    TORCH_CHECK(hidden.is_cuda(), "Input must be CUDA tensor");

    int T = hidden.size(0);
    int H = hidden.size(1);
    int B = block_tables.size(1);

    constexpr int THREADS = 128;

    int token_ctas = (T + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    int grid = 1 + token_ctas;  // last block deals with small tensors

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    preprocess_fused_cuda<TOKENS_PER_BLOCK>
        <<<grid, THREADS, 0, stream>>>(
            (const bfloat16_t*)hidden.data_ptr<at::BFloat16>(),
            block_tables.data_ptr<int>(),
            positions.data_ptr<long>(),
            slot_mapping.data_ptr<long>(),
            seq_lens.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            seq_start_loc.data_ptr<int>(),
            block_tables.stride(0),

            (bfloat16_t*)out_hidden.data_ptr<at::BFloat16>(),
            out_block_tables.data_ptr<int>(),
            out_positions.data_ptr<long>(),
            out_slot_mapping.data_ptr<long>(),
            out_seq_lens.data_ptr<int>(),
            out_context_lens.data_ptr<int>(),
            out_seq_start_loc.data_ptr<int>(),
            out_block_tables.stride(0),

            T, H, B, padded_batch_size
        );
}

void cuda_graph_preprocess_fused_dispatch(
    torch::Tensor hidden,
    torch::Tensor positions,
    torch::Tensor block_tables,
    torch::Tensor slot_mapping,
    torch::Tensor seq_lens,
    torch::Tensor context_lens,
    torch::Tensor seq_start_loc,

    torch::Tensor out_hidden,
    torch::Tensor out_positions,
    torch::Tensor out_block_tables,
    torch::Tensor out_slot_mapping,
    torch::Tensor out_seq_lens,
    torch::Tensor out_context_lens,
    torch::Tensor out_seq_start_loc,

    int64_t padded_batch_size,
    int64_t tokens_per_block
){
    switch(tokens_per_block) {
        case 1:
            launch_preprocess_fused_cuda<1>(
                hidden, block_tables, positions, slot_mapping,
                seq_lens, context_lens, seq_start_loc,
                out_hidden, out_block_tables, out_positions,
                out_slot_mapping, out_seq_lens, out_context_lens,
                out_seq_start_loc,
                (int) padded_batch_size
            );
            break;
        case 2:
            launch_preprocess_fused_cuda<2>(
                hidden, block_tables, positions, slot_mapping,
                seq_lens, context_lens, seq_start_loc,
                out_hidden, out_block_tables, out_positions,
                out_slot_mapping, out_seq_lens, out_context_lens,
                out_seq_start_loc,
                (int) padded_batch_size
            );
            break;
        case 4:
            launch_preprocess_fused_cuda<4>(
                hidden, block_tables, positions, slot_mapping,
                seq_lens, context_lens, seq_start_loc,
                out_hidden, out_block_tables, out_positions,
                out_slot_mapping, out_seq_lens, out_context_lens,
                out_seq_start_loc,
                (int) padded_batch_size
            );
            break;
        default:
            TORCH_CHECK(false, "Unsupported tokens_per_block");
    }
}


template<int TOKENS_PER_BLOCK>
__global__ void copy_graph_results_fused_kernel(
    const bfloat16_t* __restrict__ tokens,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,

    bfloat16_t* __restrict__ out_tokens,
    int* __restrict__ out_topk_ids,
    float* __restrict__ out_topk_weights,

    int num_tokens,
    int hidden_size,
    int topk
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    if (block_id == num_blocks - 1) {
        // Last block deals with topk results
        #pragma unroll
        int nelems = num_tokens * topk;
        for (int i = thread_id; i < nelems; i += num_threads) {
            out_topk_weights[i] = topk_weights[i];
            out_topk_ids[i] = topk_ids[i];
        }
    } else {
        // Other blocks deal with data tensors and block tables
        using hidden_vec_t = float4;
        constexpr int VEC_SIZE_HIDDEN = sizeof(hidden_vec_t) / sizeof(bfloat16_t);
        int start_token = block_id * TOKENS_PER_BLOCK;
        int end_token = min(start_token + TOKENS_PER_BLOCK, num_tokens);
        for (int i = start_token; i < end_token; i++) {
            const bfloat16_t* src_hidden = tokens + i * hidden_size;
            bfloat16_t* dst_hidden       = out_tokens + i * hidden_size;
            #pragma unroll
            for (int j = thread_id * VEC_SIZE_HIDDEN; j < hidden_size; j += num_threads * VEC_SIZE_HIDDEN) {
                int offset = j / VEC_SIZE_HIDDEN;
                hidden_vec_t v = reinterpret_cast<const hidden_vec_t*>(src_hidden)[offset];
                reinterpret_cast<hidden_vec_t*>(dst_hidden)[offset] = v;
            }
        }
    }
}

void copy_graph_results_fused_dispatch(
    torch::Tensor tokens,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,

    torch::Tensor out_tokens,
    torch::Tensor out_topk_ids,
    torch::Tensor out_topk_weights,

    int64_t num_tokens = 0
) {
    // only move num_tokens tokens, others are ignored
    TORCH_CHECK(tokens.size(0) >= num_tokens, "tokens must have at least num_tokens tokens");
    TORCH_CHECK(out_tokens.size(0) >= num_tokens, "out_tokens must have at least num_tokens tokens");
    TORCH_CHECK(topk_ids.size(1) == topk_weights.size(1), "topk_ids and topk_weights must have the same number of columns");

    if (num_tokens == 0) {
        num_tokens = tokens.size(0);
        TORCH_CHECK(out_tokens.size(0) == num_tokens, "out_tokens must have the same number of tokens as tokens");
        TORCH_CHECK(out_topk_ids.size(0) == num_tokens, "out_topk_ids must have the same number of tokens as tokens");
        TORCH_CHECK(out_topk_weights.size(0) == num_tokens, "out_topk_weights must have the same number of tokens as tokens");
    }
    int hidden_size = tokens.size(1);
    int topk = topk_ids.size(1);

    constexpr int TOKENS_PER_BLOCK = 2;
    constexpr int NUM_THREADS = 128;

    int token_ctas = (num_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    int grid = 1 + token_ctas;  // last block deals with small tensors

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    copy_graph_results_fused_kernel<TOKENS_PER_BLOCK>
        <<<grid, NUM_THREADS, 0, stream>>>(
            (const bfloat16_t*)tokens.data_ptr<at::BFloat16>(),
            (const int*)topk_ids.data_ptr<int>(),
            (const float*)topk_weights.data_ptr<float>(),

            (bfloat16_t*)out_tokens.data_ptr<at::BFloat16>(),
            (int*)out_topk_ids.data_ptr<int>(),
            (float*)out_topk_weights.data_ptr<float>(),
            num_tokens, hidden_size, topk
        );
}

TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def(R"(
        cuda_graph_preprocess_fused(
            Tensor hidden, Tensor positions, Tensor block_tables, Tensor slot_mapping, 
            Tensor seq_lens, Tensor context_lens, Tensor seq_start_loc, 
            Tensor out_hidden, Tensor out_positions, Tensor out_block_tables, Tensor out_slot_mapping, 
            Tensor out_seq_lens, Tensor out_context_lens, Tensor out_seq_start_loc, 
            int padded_batch_size, int tokens_per_block
        ) -> ()
    )");
    m.impl("cuda_graph_preprocess_fused", torch::kCUDA, cuda_graph_preprocess_fused_dispatch);

    m.def(R"(
        copy_graph_results_fused(
            Tensor tokens, Tensor topk_ids, Tensor topk_weights,
            Tensor out_tokens, Tensor out_topk_ids, Tensor out_topk_weights,
            int num_tokens = 0
        ) -> ()
    )");
    m.impl("copy_graph_results_fused", torch::kCUDA, copy_graph_results_fused_dispatch);
}