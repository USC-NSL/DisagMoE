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
#include "tensor_utils.hpp"

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

constexpr int MAX_GATHER_TOKENS = 1024 * 16;

// Global GDR contexts for source pointers (shared with permute.cu pattern)
gdr_context_t gather_and_sum_src_ptrs_gdr = nullptr;
gdr_context_t gather_and_sum_src_ptrs_gdr_alt = nullptr;

gdr_context_t get_gather_and_sum_src_ptrs_gdr() {
    static int enter_count = 0;
    if (gather_and_sum_src_ptrs_gdr == nullptr) {
        auto src_tensor = get_cuda_aligned_tensor(MAX_GATHER_TOKENS, torch::kUInt64);
        gather_and_sum_src_ptrs_gdr = std::make_shared<GdrContext>(src_tensor);
    }
    if (gather_and_sum_src_ptrs_gdr_alt == nullptr) {
        auto src_tensor = get_cuda_aligned_tensor(MAX_GATHER_TOKENS, torch::kUInt64);
        gather_and_sum_src_ptrs_gdr_alt = std::make_shared<GdrContext>(src_tensor);
    }
    enter_count = (enter_count + 1) & 1;
    if (enter_count & 1) {
        return gather_and_sum_src_ptrs_gdr;
    } else {
        return gather_and_sum_src_ptrs_gdr_alt;
    }
}

// Fused gather and sum kernel
// For each output token (sequence), gather topk tokens and sum them
// Each block handles one output token and one chunk of hidden_size
template <class T, int CHUNK_SIZE>
__global__ void gather_and_sum_tokens_kernel(
    T *d_out, 
    uintptr_t *d_in_ptr, 
    const int n,
    const int topk,
    const int hidden_size
) {
    int output_token_id = blockIdx.x;  // Which of the n output tokens
    int chunk_id = blockIdx.y;         // Which chunk of hidden_size
    
    if (output_token_id >= n) return;
    
    bf16 *out = reinterpret_cast<bf16 *>(d_out + output_token_id * hidden_size);
    int chunk_base = chunk_id * CHUNK_SIZE;
    bf16 *out_chunk = out + chunk_base;
    
    constexpr int WARPSIZE = 32;
    int num_warps = blockDim.x / WARPSIZE;
    int tid = threadIdx.x;
    int id_in_warp = tid % WARPSIZE;
    int wid = tid / WARPSIZE;
    
    using VEC = float2;
    constexpr int VEC_SIZE = sizeof(VEC) / sizeof(bf16);
    
    int task_per_warp = CHUNK_SIZE / num_warps / VEC_SIZE;
    int warp_base = wid * task_per_warp;
    
    // Get pointer to first token for this output
    int first_token_idx = output_token_id * topk;
    
    // Initialize output with first token
    bf16 *first_src = reinterpret_cast<bf16 *>(d_in_ptr[first_token_idx]);
    VEC *first_src_vec = (VEC *)(first_src + chunk_base);
    VEC *out_vec = (VEC *)(out_chunk);
    
    #pragma unroll
    for (int i = id_in_warp; i < task_per_warp; i += WARPSIZE) {
        out_vec[warp_base + i] = first_src_vec[warp_base + i];
    }
    
    // Accumulate remaining topk-1 tokens
    for (int k = 1; k < topk; k++) {
        int token_idx = first_token_idx + k;
        bf16 *src = reinterpret_cast<bf16 *>(d_in_ptr[token_idx]);
        VEC *src_vec = (VEC *)(src + chunk_base);
        
        #pragma unroll
        for (int i = id_in_warp; i < task_per_warp; i += WARPSIZE) {
            VEC src_val = src_vec[warp_base + i];
            VEC acc_val = out_vec[warp_base + i];
            
            // Use bf162 addition directly (each float in float2 contains 2 bf16 values as bf162)
            bf162 *src_x = reinterpret_cast<bf162 *>(&src_val.x);
            bf162 *src_y = reinterpret_cast<bf162 *>(&src_val.y);
            bf162 *acc_x = reinterpret_cast<bf162 *>(&acc_val.x);
            bf162 *acc_y = reinterpret_cast<bf162 *>(&acc_val.y);
            
            *acc_x = __hadd2(*src_x, *acc_x);
            *acc_y = __hadd2(*src_y, *acc_y);
            
            out_vec[warp_base + i] = acc_val;
        }
    }
}

#define LAUNCH_GATHER_AND_SUM_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(n, hidden_size / chunk_size, 1); \
    gather_and_sum_tokens_kernel<T, chunk_size><<<grid, block, 0, stream>>>(dest, src_ptr, n, topk, hidden_size); \
} while(0)

template <class T>
void _gather_and_sum_tokens_cuda(
    T *dest, 
    uintptr_t *src_ptr, 
    int n, 
    int topk, 
    int hidden_size
) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size > 0);
    constexpr int num_threads = 128;
    dim3 block(num_threads, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (hidden_size % 2048 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(2048);
    } else if (hidden_size % 960 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(960);
    } else if (hidden_size % 512 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(512);
    } else if (hidden_size % 256 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(256);
    } else {
        LAUNCH_GATHER_AND_SUM_KERNEL_(128);
    }
}
void gather_and_sum_tokens_cuda_dispatch(
    torch::Tensor dest, 
    int64_t src_ptr, 
    int64_t n, 
    int64_t topk, 
    int64_t hidden_size
) {
    // dest is a cuda ptr with shape [n, hidden_size]
    // src_ptr is a cpu ptr to array of n*topk uintptr_t pointers
    uintptr_t* src_ptr_host = reinterpret_cast<uintptr_t*>(src_ptr);
    using scalar_t = c10::BFloat16;
    
#if KERNEL_USE_GDRCOPY == 1
    gdr_context_t gather_src_ptrs_gdr = get_gather_and_sum_src_ptrs_gdr();
    gather_src_ptrs_gdr->copy_from_host(src_ptr_host, n * topk * sizeof(uintptr_t));
    auto src_tensor = gather_src_ptrs_gdr->get_tensor();
    src_tensor = src_tensor.narrow(0, 0, n * topk);
    _gather_and_sum_tokens_cuda<scalar_t>(
        dest.data_ptr<scalar_t>(), 
        src_tensor.data_ptr<uintptr_t>(), 
        n, 
        topk, 
        hidden_size
    );
#else
    // Create a torch tensor and copy from host
    auto src_tensor = torch::empty({n * topk}, torch::TensorOptions()
        .dtype(torch::kUInt64)
        .device(torch::kCUDA));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDACHECK(cudaMemcpyAsync(src_tensor.data_ptr<uintptr_t>(), src_ptr_host, 
               n * topk * sizeof(uintptr_t), cudaMemcpyHostToDevice, stream));
    _gather_and_sum_tokens_cuda<scalar_t>(
        dest.data_ptr<scalar_t>(), 
        src_tensor.data_ptr<uintptr_t>(), 
        n, 
        topk, 
        hidden_size
    );
#endif
}

TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def("gather_and_sum_tokens(Tensor dest, int src_ptr, int n, int topk, int hidden_size) -> ()");
    m.impl("gather_and_sum_tokens", torch::kCUDA, gather_and_sum_tokens_cuda_dispatch);
}
