#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

using namespace at;

using bfloat16_t = __nv_bfloat16;

template<int TOKENS_PER_BLOCK>
__global__ void copy_and_pad_kernel(
    const bfloat16_t* __restrict__ in_hiddens,
    const long* __restrict__ in_batch_sizes,
    const int* __restrict__ in_m_indices,
    bfloat16_t* __restrict__ out_hiddens,
    long* __restrict__ out_batch_sizes,
    int* __restrict__ out_m_indices,
    int num_tokens, int num_experts, 
    int padded_bsz, int hidden_size
) {
    int num_blocks = gridDim.x;
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    if (block_id == num_blocks - 1) {
        // Last block deals with batch sizes and m_indices
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads) {
            out_batch_sizes[i] = in_batch_sizes[i];
        }

        #pragma unroll
        for (int i = thread_id; i < padded_bsz; i += num_threads) {
            if (i < num_tokens) {
                out_m_indices[i] = in_m_indices[i];
            } else {
                out_m_indices[i] = 0;
            }
        }

    } else {
        // Other blocks deal with hidden states
        using hidden_vec_t = float4;
        constexpr int VEC_SIZE_HIDDEN = sizeof(hidden_vec_t) / sizeof(bfloat16_t);

        int start_row = block_id * TOKENS_PER_BLOCK;

        for (int r = 0; r < TOKENS_PER_BLOCK; r++) {
            int row = start_row + r;
            if (row >= num_tokens) return;

            const bfloat16_t* src_hidden = in_hiddens + row * hidden_size;
            bfloat16_t* dst_hidden = out_hiddens + row * hidden_size;

            int vec_end = (hidden_size / VEC_SIZE_HIDDEN) * VEC_SIZE_HIDDEN;

            #pragma unroll
            for (int i = thread_id * VEC_SIZE_HIDDEN; i < vec_end; i += num_threads * VEC_SIZE_HIDDEN) {
                int offset = i / VEC_SIZE_HIDDEN;
                hidden_vec_t v = reinterpret_cast<const hidden_vec_t*>(src_hidden)[offset];
                reinterpret_cast<hidden_vec_t*>(dst_hidden)[offset] = v;
            }
        }
    }
}

template<int TOKENS_PER_BLOCK>
void launch_copy_and_pad_cuda(
    const at::Tensor& in_hiddens,
    const at::Tensor& in_batch_sizes,
    const at::Tensor& in_m_indices,
    at::Tensor& out_hiddens,
    at::Tensor& out_batch_sizes,
    at::Tensor& out_m_indices,
    int padded_bsz
) {
    TORCH_CHECK(in_hiddens.is_cuda(), "Input must be CUDA tensor");

    int num_tokens = in_hiddens.size(0);
    int num_experts = in_batch_sizes.size(0);
    int hidden_size = in_hiddens.size(1);

    constexpr int THREADS = 128;

    int num_ctas = (num_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    int grid = 1 + num_ctas;  // last block deals with m_indices

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    copy_and_pad_kernel<TOKENS_PER_BLOCK>
        <<<grid, THREADS, 0, stream>>>(
            (const bfloat16_t*)in_hiddens.data_ptr<at::BFloat16>(),
            in_batch_sizes.data_ptr<long>(),
            in_m_indices.data_ptr<int>(),
            (bfloat16_t*)out_hiddens.data_ptr<at::BFloat16>(),
            out_batch_sizes.data_ptr<long>(),
            out_m_indices.data_ptr<int>(),
            num_tokens, num_experts, padded_bsz, hidden_size
        );
}

void fused_copy_and_pad_dispatch(
    torch::Tensor in_hiddens,
    torch::Tensor in_batch_sizes,
    torch::Tensor in_m_indices,
    torch::Tensor out_hiddens,
    torch::Tensor out_batch_sizes,
    torch::Tensor out_m_indices,
    int64_t padded_bsz,
    int64_t tokens_per_block
) {
    switch(tokens_per_block) {
        case 1:
            launch_copy_and_pad_cuda<1>(
                in_hiddens, in_batch_sizes, in_m_indices,
                out_hiddens, out_batch_sizes, out_m_indices,
                (int)padded_bsz
            );
            break;
        case 2:
            launch_copy_and_pad_cuda<2>(
                in_hiddens, in_batch_sizes, in_m_indices,
                out_hiddens, out_batch_sizes, out_m_indices,
                (int)padded_bsz
            );
            break;
        case 4:
            launch_copy_and_pad_cuda<4>(
                in_hiddens, in_batch_sizes, in_m_indices,
                out_hiddens, out_batch_sizes, out_m_indices,
                (int)padded_bsz
            );
            break;
        default:
            TORCH_CHECK(false, "Unsupported tokens_per_block");
    }
}

// Register the operator
TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def(R"(
        fused_copy_and_pad(
            Tensor in_hiddens, Tensor in_batch_sizes, Tensor in_m_indices, 
            Tensor out_hiddens, Tensor out_batch_sizes, Tensor out_m_indices, 
            int padded_bsz, int tokens_per_block
        ) -> ()
    )");
    m.impl("fused_copy_and_pad", torch::kCUDA, fused_copy_and_pad_dispatch);
}
