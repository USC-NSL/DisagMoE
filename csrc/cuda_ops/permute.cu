#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <assert.h>
#include <cstring>
#include <string>
#include <vector>

#include "permute.h"

template <class T, int CHUNK_SIZE>
__device__ void move_one_token_kernel(T *dest, T *src, const int hidden_size) {
    constexpr int WARPSIZE = 32;

    int chunk_id = blockIdx.y;
    int num_warps = blockDim.x / WARPSIZE;

    int tid = threadIdx.x;
    int id_in_warp = tid % WARPSIZE;
    int wid = tid / WARPSIZE;

    int chunk_base = chunk_id * CHUNK_SIZE;
    
    using VEC = __half;
    if constexpr (CHUNK_SIZE == 1024) {
        using VEC = __half2;
    } else if constexpr (CHUNK_SIZE == 2048) {
        using VEC = float2;
    } else {
        using VEC = float4;
    }
    constexpr int VEC_SIZE = sizeof(VEC) / sizeof(T);

    VEC *src_vec = (VEC *)(src + chunk_base);
    VEC *dest_vec = (VEC *)(dest + chunk_base);

    int task_per_warp = CHUNK_SIZE / num_warps / VEC_SIZE;
    int warp_base = wid * task_per_warp;

    #pragma unroll
    for (int i = id_in_warp; i < task_per_warp; i += WARPSIZE) {
        dest_vec[warp_base + i] = src_vec[warp_base + i];
    }
}

template <class T, int CHUNK_SIZE>
__global__ void permute_tokens_kernel(T *d_out, T *d_in, int *mappings, const int hidden_size) {
    int token_id = blockIdx.x;
    int p = mappings[token_id];
    move_one_token_kernel<T, CHUNK_SIZE>(d_out + p * hidden_size, d_in + token_id * hidden_size, hidden_size);
}

#define LAUNCH_PERMUTE_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(num_tokens, hidden_size / chunk_size, 1); \
    permute_tokens_kernel<T, chunk_size><<<grid, block>>>(dest, src, mappings, hidden_size); \
} while(0)
    
template <class T>
void _permute_tokens_cuda(T *dest, T *src, int *mappings, int num_tokens, int hidden_size) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size >= 2048 && hidden_size % 2048 == 0);
    constexpr int num_threads = 128;
    dim3 block(num_threads, 1, 1);
    if (num_tokens <= 80) {
        LAUNCH_PERMUTE_KERNEL_(512);
    } else if (num_tokens <= 160) {
        LAUNCH_PERMUTE_KERNEL_(1024);
    } else {
        LAUNCH_PERMUTE_KERNEL_(2048);
    }
}

torch::Tensor permute_tokens_cuda(torch::Tensor tokens, torch::Tensor mappings) {
    assert(tokens.dim() == 2);
    assert(mappings.dim() == 1);
    assert(tokens.size(0) == mappings.size(0));

    int num_tokens = tokens.size(0);
    int hidden_size = tokens.size(1);

    torch::Tensor out = torch::empty_like(tokens);
   
    AT_DISPATCH_REDUCED_FLOATING_TYPES(tokens.scalar_type(), "permute_tokens_cuda", [&] {
        _permute_tokens_cuda<scalar_t>(out.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), mappings.data_ptr<int>(), num_tokens, hidden_size);
    });

    return out;
}

template <class T, int CHUNK_SIZE>
__global__ void gather_tokens_kernel(T *d_out, uintptr_t *d_in_ptr, const int hidden_size) {
    int token_id = blockIdx.x;
    T *d_in = (T *)d_in_ptr[token_id];
    move_one_token_kernel<T, CHUNK_SIZE>(d_out + token_id * hidden_size, d_in, hidden_size);
}


#define LAUNCH_GATHER_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(num_tokens, hidden_size / chunk_size, 1); \
    gather_tokens_kernel<T, chunk_size><<<grid, block, 0, stream>>>(dest, src_ptr, hidden_size); \
} while(0)

template <class T>
void _gather_tokens_cuda(T *dest, uintptr_t *src_ptr, int num_tokens, int hidden_size, cudaStream_t stream) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size >= 2048 && hidden_size % 2048 == 0);
    constexpr int num_threads = 128;
    dim3 block(num_threads, 1, 1);
    if (num_tokens <= 80) {
        LAUNCH_GATHER_KERNEL_(512);
    } else if (num_tokens <= 160) {
        LAUNCH_GATHER_KERNEL_(1024);
    } else {
        LAUNCH_GATHER_KERNEL_(2048);
    }
}

void gather_tokens_cuda(torch::Tensor dest, uintptr_t *src_ptr, int num_tokens, int hidden_size, cudaStream_t stream) {
    // dest is a cuda ptr, src_ptr is a cpu ptr
    at::cuda::CUDAStream c10_stream = at::cuda::getCurrentCUDAStream(0);
    std::cout << "current stream: " << c10_stream.stream() << std::endl;
    torch::Tensor src_tensor = torch::from_blob(src_ptr, {num_tokens}, torch::TensorOptions().dtype(torch::kUInt64)).to(dest.device());
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dest.scalar_type(), "gather_tokens_cuda", [&] {
        _gather_tokens_cuda<scalar_t>(dest.data_ptr<scalar_t>(), src_tensor.data_ptr<uintptr_t>(), num_tokens, hidden_size, stream);
    });
    CUDACHECK(cudaStreamSynchronize(stream));
}