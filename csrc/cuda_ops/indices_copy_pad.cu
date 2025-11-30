#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include "indices_copy_pad.cuh"

__global__ void copy_and_pad_kernel(
    const uint16_t* __restrict__ in_hiddens,
    const int32_t* __restrict__ in_m_indices,
    uint16_t* __restrict__ out_hiddens,
    int32_t* __restrict__ out_m_indices,
    int bs, int bucket_bs, int hidden_size
) {
    // 1D grid of blocks handling rows.
    // blockDim.x = 16 (rows per block)
    // blockDim.y = 32 (threads per row for parallel copy)
    
    int row_in_block = threadIdx.x;
    int row = blockIdx.x * blockDim.x + row_in_block;

    if (row >= bucket_bs) return;

    bool valid = row < bs;

    // 1. Copy m_indices (One thread per row handles this)
    if (threadIdx.y == 0) {
        if (valid) {
            out_m_indices[row] = in_m_indices[row];
        } else {
            out_m_indices[row] = 0;
        }
    }

    // 2. Copy hiddens (Parallel copy across threads in y dim)
    for (int k = threadIdx.y; k < hidden_size; k += blockDim.y) {
        if (valid) {
            out_hiddens[row * hidden_size + k] = in_hiddens[row * hidden_size + k];
        } else {
            out_hiddens[row * hidden_size + k] = 0;
        }
    }
}

void fused_copy_and_pad(
    torch::Tensor in_hiddens,
    torch::Tensor in_m_indices,
    torch::Tensor out_hiddens,
    torch::Tensor out_m_indices,
    int bs, int bucket_bs
) {
    int hidden_size = in_hiddens.size(1);
    
    // Block: x=16 rows, y=32 threads/row
    dim3 threads(16, 32);
    dim3 blocks((bucket_bs + 15) / 16);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Casting bf16 (2 bytes) to uint16_t for bitwise copy
    copy_and_pad_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint16_t*>(in_hiddens.data_ptr()),
        in_m_indices.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(out_hiddens.data_ptr()),
        out_m_indices.data_ptr<int32_t>(),
        bs, bucket_bs, hidden_size
    );
}

// Register the operator
TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def(
        "fused_copy_and_pad(Tensor in_hiddens, Tensor in_m_indices, Tensor out_hiddens, Tensor out_m_indices, "
        "int bs, int bucket_bs) -> ()");
    m.impl("fused_copy_and_pad", torch::kCUDA, fused_copy_and_pad);
}
