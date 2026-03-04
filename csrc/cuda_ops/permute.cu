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
#include "cuda_utils.h"
#include "tensor_utils.hpp"

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

template <int CHUNK_SIZE, bool apply_weights = false>
__device__ void move_one_token_kernel(bf16 *dest, bf16 *src, const int hidden_size, float token_weight = .0f) {

    constexpr int WARPSIZE = 32;

    int chunk_id = blockIdx.y;
    int num_warps = blockDim.x / WARPSIZE;

    int tid = threadIdx.x;
    int id_in_warp = tid % WARPSIZE;
    int wid = tid / WARPSIZE;

    int chunk_base = chunk_id * CHUNK_SIZE;
    
    using VEC = float2;
    constexpr int VEC_SIZE = sizeof(VEC) / sizeof(bf16);

    VEC *src_vec = (VEC *)(src + chunk_base);
    VEC *dest_vec = (VEC *)(dest + chunk_base);

    int task_per_warp = CHUNK_SIZE / num_warps / VEC_SIZE;
    int warp_base = wid * task_per_warp;

    bf16 w = __float2bfloat16(token_weight);
    const bf162 w2 = {w, w};

    #pragma unroll
    for (int i = id_in_warp; i < task_per_warp; i += WARPSIZE) {
        VEC val = src_vec[warp_base + i];
        if constexpr (apply_weights) {
            bf162 *v1 = reinterpret_cast<bf162 *>(&val.x);
            *v1 = __hmul2(*v1, w2);
            bf162 *v2 = reinterpret_cast<bf162 *>(&val.y);
            *v2 = __hmul2(*v2, w2);
        }
        dest_vec[warp_base + i] = val;
    }
}

template <class T, int CHUNK_SIZE>
__global__ void permute_tokens_kernel(T *d_out, T *d_in, int *mappings, const int topk, const int hidden_size) {
    int token_id = blockIdx.x;
    int p = mappings[token_id];
    bf16 *out = reinterpret_cast<bf16 *>(d_out + p * hidden_size);
    bf16 *src = reinterpret_cast<bf16 *>(d_in + (token_id / topk) * hidden_size);
    move_one_token_kernel<CHUNK_SIZE>(out, src, hidden_size);
}

#define LAUNCH_PERMUTE_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(num_output_tokens, hidden_size / chunk_size, 1); \
    permute_tokens_kernel<T, chunk_size><<<grid, block, 0, stream>>>(dest, src, mappings, topk, hidden_size); \
} while(0)
    
template <class T>
void _permute_tokens_cuda(T *dest, T *src, int *mappings, int num_input_tokens, int num_output_tokens, int hidden_size) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size > 0);
    constexpr int num_threads = 128;
    int topk = num_output_tokens / num_input_tokens;
    dim3 block(num_threads, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (hidden_size % 2048 == 0) {
        LAUNCH_PERMUTE_KERNEL_(2048);
    } else if (hidden_size % 960 == 0) {
        LAUNCH_PERMUTE_KERNEL_(960);
    } else if (hidden_size % 512 == 0) {
        LAUNCH_PERMUTE_KERNEL_(512);
    } else if (hidden_size % 256 == 0) {
        LAUNCH_PERMUTE_KERNEL_(256);
    } else {
        LAUNCH_PERMUTE_KERNEL_(128);
    }
}

// This kernel is used to permute the tokens in the hidden states
// 1. if num of tokens equals to size of mappings, do normal permutation
// 2. if num of tokens is smaller than size of mappings, do topk token scatter
torch::Tensor permute_tokens_cuda_dispatch(torch::Tensor tokens, torch::Tensor mappings) {

    TORCH_CHECK(tokens.dim() == 2, "tokens must be a 2D tensor");
    TORCH_CHECK(mappings.dim() == 1, "mappings must be a 1D tensor");
    TORCH_CHECK(tokens.device() == mappings.device(), "tokens and mappings must be on the same device");
    TORCH_CHECK(tokens.scalar_type() == at::kBFloat16, "tokens must be a BFloat16 tensor");

    int num_input_tokens = tokens.size(0);
    int num_output_tokens = mappings.size(0);
    int hidden_size = tokens.size(1);

    TORCH_CHECK(num_output_tokens % num_input_tokens == 0, "num_output_tokens must be divisible by num_input_tokens");

    torch::Tensor out = torch::empty({num_output_tokens, hidden_size}, tokens.options());
    
    using scalar_t = c10::BFloat16;
    _permute_tokens_cuda<scalar_t>(
        out.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), mappings.data_ptr<int>(), 
        num_input_tokens, num_output_tokens, hidden_size
    );

    return out;
}

template <class T, int CHUNK_SIZE>
__global__ void apply_weights_and_permute_tokens_kernel(T *d_out, T *d_in, float *d_weights, int *mappings, const int hidden_size) {
    int token_id = blockIdx.x;
    int p = mappings[token_id];
    bf16 *out = reinterpret_cast<bf16 *>(d_out + p * hidden_size);
    bf16 *src = reinterpret_cast<bf16 *>(d_in + token_id * hidden_size);
    float *weights = reinterpret_cast<float *>(d_weights + token_id);
    move_one_token_kernel<CHUNK_SIZE>(out, src, hidden_size, weights[token_id]);
}

#define LAUNCH_APPLY_WEIGHTS_AND_PERMUTE_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(num_tokens, hidden_size / chunk_size, 1); \
    apply_weights_and_permute_tokens_kernel<T, chunk_size><<<grid, block, 0, stream>>>(dest, src, weights, mappings, hidden_size); \
} while(0)

template <class T>
void _apply_weights_and_permute_tokens_cuda(T *dest, T *src, float *weights, int *mappings, int num_tokens, int hidden_size) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size > 0);
    constexpr int num_threads = 128;
    dim3 block(num_threads, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (hidden_size % 2048 == 0) {
        LAUNCH_APPLY_WEIGHTS_AND_PERMUTE_KERNEL_(2048);
    } else if (hidden_size % 960 == 0) {
        LAUNCH_APPLY_WEIGHTS_AND_PERMUTE_KERNEL_(960);
    } else if (hidden_size % 512 == 0) {
        LAUNCH_APPLY_WEIGHTS_AND_PERMUTE_KERNEL_(512);
    } else if (hidden_size % 256 == 0) {
        LAUNCH_APPLY_WEIGHTS_AND_PERMUTE_KERNEL_(256);
    } else {
        LAUNCH_APPLY_WEIGHTS_AND_PERMUTE_KERNEL_(128);
    }
}

torch::Tensor apply_weights_and_permute_tokens_cuda_dispatch(torch::Tensor tokens, torch::Tensor weights, torch::Tensor mappings) {
    // weights and mappings can be padded, paddings are ignored
    TORCH_CHECK(tokens.dim() == 2, "tokens must be a 2D tensor");
    TORCH_CHECK(weights.dim() == 1, "weights must be a 1D tensor");
    TORCH_CHECK(mappings.dim() == 1, "mappings must be a 1D tensor");
    TORCH_CHECK(tokens.scalar_type() == at::kBFloat16, "tokens must be a BFloat16 tensor");

    int num_input_tokens = tokens.size(0);
    int num_output_tokens = mappings.size(0);
    int hidden_size = tokens.size(1);

    torch::Tensor out = torch::empty_like(tokens);

    using scalar_t = c10::BFloat16;
    _apply_weights_and_permute_tokens_cuda<scalar_t>(
        out.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), weights.data_ptr<float>(), mappings.data_ptr<int>(), 
        num_input_tokens, hidden_size
    );

    return out;
}

template <class T, int CHUNK_SIZE>
__global__ void gather_tokens_kernel(T *d_out, uintptr_t *d_in_ptr, const int hidden_size) {
    int token_id = blockIdx.x;
    bf16 *out = reinterpret_cast<bf16 *>(d_out + token_id * hidden_size);
    bf16 *src = reinterpret_cast<bf16 *>(d_in_ptr[token_id]);
    move_one_token_kernel<CHUNK_SIZE>(out, src, hidden_size);
}

#define LAUNCH_GATHER_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(num_tokens, hidden_size / chunk_size, 1); \
    gather_tokens_kernel<T, chunk_size><<<grid, block, 0, stream>>>(dest, src_ptr, hidden_size); \
} while(0)

template <class T>
void _gather_tokens_cuda(T *dest, uintptr_t *src_ptr, int num_tokens, int hidden_size) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size > 0);
    constexpr int num_threads = 128;
    dim3 block(num_threads, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (hidden_size % 2048 == 0) {
        LAUNCH_GATHER_KERNEL_(2048);
    } else if (hidden_size % 960 == 0) {
        LAUNCH_GATHER_KERNEL_(960);
    } else if (hidden_size % 512 == 0) {
        LAUNCH_GATHER_KERNEL_(512);
    } else if (hidden_size % 256 == 0) {
        LAUNCH_GATHER_KERNEL_(256);
    } else {
        LAUNCH_GATHER_KERNEL_(128);
    }
}

constexpr int MAX_GATHER_TOKENS = 1024 * 16;
gdr_context_t gather_src_ptrs_gdr = nullptr;
gdr_context_t gather_src_ptrs_gdr_alt = nullptr;

gdr_context_t get_gather_src_ptrs_gdr() {
    static int enter_count = 0;
    if (gather_src_ptrs_gdr == nullptr) {
        auto src_tensor = get_cuda_aligned_tensor(MAX_GATHER_TOKENS, torch::kUInt64);
        gather_src_ptrs_gdr = std::make_shared<GdrContext>(src_tensor);
    }
    if (gather_src_ptrs_gdr_alt == nullptr) {
        auto src_tensor = get_cuda_aligned_tensor(MAX_GATHER_TOKENS, torch::kUInt64);
        gather_src_ptrs_gdr_alt = std::make_shared<GdrContext>(src_tensor);
    }
    enter_count = (enter_count + 1) & 1;
    if (enter_count & 1) {
        return gather_src_ptrs_gdr;
    } else {
        return gather_src_ptrs_gdr_alt;
    }
}

void gather_tokens_cuda_dispatch(torch::Tensor dest, int64_t src_ptr, int64_t num_tokens, int64_t hidden_size) {
    // dest is a cuda ptr, src_ptr is a cpu ptr
    uintptr_t* src_ptr_host = reinterpret_cast<uintptr_t*>(src_ptr);
    using scalar_t = c10::BFloat16;
#if KERNEL_USE_GDRCOPY == 1
    gdr_context_t gather_src_ptrs_gdr = get_gather_src_ptrs_gdr();
    gather_src_ptrs_gdr->copy_from_host(src_ptr_host, num_tokens * sizeof(uintptr_t));
    auto src_tensor = gather_src_ptrs_gdr->get_tensor();
    src_tensor = src_tensor.narrow(0, 0, num_tokens);
    _gather_tokens_cuda<scalar_t>(dest.data_ptr<scalar_t>(), src_tensor.data_ptr<uintptr_t>(), num_tokens, hidden_size);
#else
    // Create a torch tensor and copy from host
    auto src_tensor = torch::empty({num_tokens}, torch::TensorOptions()
        .dtype(torch::kUInt64)
        .device(torch::kCUDA));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDACHECK(cudaMemcpyAsync(src_tensor.data_ptr<uintptr_t>(), src_ptr_host, 
               num_tokens * sizeof(uintptr_t), cudaMemcpyHostToDevice, stream));
    _gather_tokens_cuda<scalar_t>(dest.data_ptr<scalar_t>(), src_tensor.data_ptr<uintptr_t>(), num_tokens, hidden_size);
#endif
}

TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def("permute_tokens(Tensor tokens, Tensor mappings) -> Tensor");
    m.impl("permute_tokens", torch::kCUDA, permute_tokens_cuda_dispatch);

    m.def("gather_tokens(Tensor dest, int src_ptr, int num_tokens, int hidden_size) -> ()");
    m.impl("gather_tokens", torch::kCUDA, gather_tokens_cuda_dispatch);

    m.def("apply_weights_and_permute_tokens(Tensor tokens, Tensor weights, Tensor mappings) -> Tensor");
    m.impl("apply_weights_and_permute_tokens", torch::kCUDA, apply_weights_and_permute_tokens_cuda_dispatch);
}
