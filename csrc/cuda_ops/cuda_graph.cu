#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "cuda_graph.h"

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

    int T, int H, int B
){
    int num_blocks = gridDim.x;
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    if (block_id == num_blocks - 1) {
        // Last block deals with metadata tensors
        #pragma unroll
        for (int i = thread_id; i < T; i += num_threads) {
            out_positions[i]    = positions[i];
            out_slot_mapping[i] = slot_mapping[i];
            out_seq_lens[i]     = seq_lens[i];
            out_context_lens[i] = context_lens[i];
        }
        #pragma unroll
        for (int i = thread_id; i < T+1; i += num_threads) {
            out_seq_start_loc[i] = seq_start_loc[i];
        }
    } else {
        // Other blocks deal with data tensors and block tables
        constexpr int VEC_SIZE_HIDDEN = 8;   // float4 = 16 bytes = 8 fp16
        constexpr int VEC_SIZE_BLOCK_TABLE = 4;
        int start_token = block_id * TOKENS_PER_BLOCK;

        for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
            int token = start_token + t;
            if (token >= T) return;

            const bfloat16_t* src_hidden = hidden + token * H;
            bfloat16_t* dst_hidden       = out_hidden + token * H;

            #pragma unroll
            for (int i = thread_id * VEC_SIZE_HIDDEN; i < H; i += num_threads * VEC_SIZE_HIDDEN) {
                int offset = i / VEC_SIZE_HIDDEN;
                float4 v = reinterpret_cast<const float4*>(src_hidden)[offset];
                reinterpret_cast<float4*>(dst_hidden)[offset] = v;
            }

            const int* src_bt = block_tables + token * in_block_table_stride;
            int* dst_bt       = out_block_tables + token * out_block_table_stride;

            int vec_end = (B / VEC_SIZE_BLOCK_TABLE) * VEC_SIZE_BLOCK_TABLE; 

            #pragma unroll
            for (int i = thread_id * VEC_SIZE_BLOCK_TABLE; i < vec_end; i += num_threads * VEC_SIZE_BLOCK_TABLE) {
                int offset = i / VEC_SIZE_BLOCK_TABLE;
                int4 v = reinterpret_cast<const int4*>(src_bt)[offset];
                reinterpret_cast<int4*>(dst_bt)[offset] = v;
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

    cudaStream_t stream
){
    TORCH_CHECK(hidden.is_cuda(), "Input must be CUDA tensor");

    int T = hidden.size(0);
    int H = hidden.size(1);
    int B = block_tables.size(1);

    constexpr int THREADS = 128;

    int token_ctas = (T + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    int grid = 1 + token_ctas;  // last block deals with small tensors

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

            T, H, B
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

    int64_t tokens_per_block,
    int64_t raw_cuda_stream
){
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(raw_cuda_stream);
    switch(tokens_per_block) {
        case 1:
            launch_preprocess_fused_cuda<1>(
                hidden, block_tables, positions, slot_mapping,
                seq_lens, context_lens, seq_start_loc,
                out_hidden, out_block_tables, out_positions,
                out_slot_mapping, out_seq_lens, out_context_lens,
                out_seq_start_loc, stream
            );
            break;
        case 2:
            launch_preprocess_fused_cuda<2>(
                hidden, block_tables, positions, slot_mapping,
                seq_lens, context_lens, seq_start_loc,
                out_hidden, out_block_tables, out_positions,
                out_slot_mapping, out_seq_lens, out_context_lens,
                out_seq_start_loc, stream
            );
            break;
        case 4:
            launch_preprocess_fused_cuda<4>(
                hidden, block_tables, positions, slot_mapping,
                seq_lens, context_lens, seq_start_loc,
                out_hidden, out_block_tables, out_positions,
                out_slot_mapping, out_seq_lens, out_context_lens,
                out_seq_start_loc, stream
            );
            break;
        default:
            TORCH_CHECK(false, "Unsupported tokens_per_block");
    }
}

TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def(R"(
        cuda_graph_preprocess_fused(
            Tensor hidden, Tensor positions, Tensor block_tables, Tensor slot_mapping, 
            Tensor seq_lens, Tensor context_lens, Tensor seq_start_loc, 
            Tensor out_hidden, Tensor out_positions, Tensor out_block_tables, Tensor out_slot_mapping, 
            Tensor out_seq_lens, Tensor out_context_lens, Tensor out_seq_start_loc, 
            int tokens_per_block, int raw_cuda_stream
        ) -> void
    )");
    m.impl("cuda_graph_preprocess_fused", torch::kCUDA, cuda_graph_preprocess_fused_dispatch);
}