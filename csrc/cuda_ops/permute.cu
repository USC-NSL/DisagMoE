#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <assert.h>
#include <cstring>
#include <memory>

#include "gdr_context.hpp"
#include "permute.h"
#include "cuda_utils.h"
#include "tensor_utils.hpp"

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
__global__ void permute_tokens_kernel(T *d_out, T *d_in, int *mappings, const int topk, const int hidden_size) {
    int token_id = blockIdx.x;
    int p = mappings[token_id];
    move_one_token_kernel<T, CHUNK_SIZE>(d_out + p * hidden_size, d_in + (token_id / topk) * hidden_size, hidden_size);
}

#define LAUNCH_PERMUTE_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(num_output_tokens, hidden_size / chunk_size, 1); \
    permute_tokens_kernel<T, chunk_size><<<grid, block>>>(dest, src, mappings, topk, hidden_size); \
} while(0)
    
template <class T>
void _permute_tokens_cuda(T *dest, T *src, int *mappings, int num_input_tokens, int num_output_tokens, int hidden_size) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size >= 2048 && hidden_size % 2048 == 0);
    constexpr int num_threads = 128;
    int topk = num_output_tokens / num_input_tokens;
    dim3 block(num_threads, 1, 1);
    if (num_output_tokens <= 80) {
        LAUNCH_PERMUTE_KERNEL_(512);
    } else if (num_output_tokens <= 160) {
        LAUNCH_PERMUTE_KERNEL_(1024);
    } else {
        LAUNCH_PERMUTE_KERNEL_(2048);
    }
}

// This kernel is used to permute the tokens in the hidden states
// 1. if num of tokens equals to size of mappings, do normal permutation
// 2. if num of tokens is smaller than size of mappings, do topk token scatter
torch::Tensor permute_tokens_cuda_dispatch(torch::Tensor tokens, torch::Tensor mappings) {

    assert(tokens.dim() == 2);
    assert(mappings.dim() == 1);

    int num_input_tokens = tokens.size(0);
    int num_output_tokens = mappings.size(0);
    int hidden_size = tokens.size(1);

    assert(num_output_tokens % num_input_tokens == 0);

    torch::Tensor out = torch::empty({num_output_tokens, hidden_size}, tokens.options());

    // torch::cuda::synchronize(); // "cuda illegal memory access" without this line, seems to in a different stream with pytorch
   
    AT_DISPATCH_REDUCED_FLOATING_TYPES(tokens.scalar_type(), "permute_tokens_cuda", [&] {
        _permute_tokens_cuda<scalar_t>(
            out.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), mappings.data_ptr<int>(), 
            num_input_tokens, num_output_tokens, hidden_size
        );
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


constexpr int MAX_GATHER_TOKENS = 1024 * 16;
gdr_context_t gather_src_ptrs_gdr = nullptr;

gdr_context_t get_gather_src_ptrs_gdr() {
    if (gather_src_ptrs_gdr == nullptr) {
        auto src_tensor = get_cuda_aligned_tensor(MAX_GATHER_TOKENS, torch::kUInt64);
        gather_src_ptrs_gdr = std::make_shared<GdrContext>(src_tensor);
    }
    return gather_src_ptrs_gdr;
}

void gather_tokens_cuda_dispatch(torch::Tensor dest, int64_t src_ptr, int64_t num_tokens, int64_t hidden_size, int64_t raw_cuda_stream) {
    // dest is a cuda ptr, src_ptr is a cpu ptr
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(raw_cuda_stream);
    uintptr_t* src_ptr_host = reinterpret_cast<uintptr_t*>(src_ptr);
    gdr_context_t gather_src_ptrs_gdr = get_gather_src_ptrs_gdr();
    gather_src_ptrs_gdr->copy_from_host(src_ptr_host, num_tokens * sizeof(uintptr_t));
    auto src_tensor = gather_src_ptrs_gdr->get_tensor();
    src_tensor = src_tensor.narrow(0, 0, num_tokens);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dest.scalar_type(), "gather_tokens_cuda", [&] {
        _gather_tokens_cuda<scalar_t>(dest.data_ptr<scalar_t>(), src_tensor.data_ptr<uintptr_t>(), num_tokens, hidden_size, stream);
    });
}

TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def("permute_tokens(Tensor tokens, Tensor mappings) -> Tensor");
    m.impl("permute_tokens", torch::kCUDA, permute_tokens_cuda_dispatch);

    m.def("gather_tokens(Tensor dest, int src_ptr, int num_tokens, int hidden_size, int stream) -> ()");
    m.impl("gather_tokens", torch::kCUDA, gather_tokens_cuda_dispatch);
}
