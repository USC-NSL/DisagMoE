#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <string>

namespace disagmoe {

/// One-time initialization: detects hardware and selects optimal CUTLASS tile
/// configuration.  Must be called before any grouped_gemm() calls.
///
/// @param device_id   CUDA device ordinal
/// @return  Human-readable description of the selected config (for logging).
std::string init_grouped_gemm(int device_id);

/// Grouped GEMM: C[i] = A_slice[i] @ B[i]  for each expert i.
///
/// @param a       [total_tokens, K]      bf16, packed across experts
/// @param b       [num_experts, K, N]    bf16 weights
/// @param c       [total_tokens, N]      bf16 output (pre-allocated)
/// @param batch_sizes  [num_experts]     int64, number of tokens per expert (CPU or CUDA)
void grouped_gemm(torch::Tensor a,
                  torch::Tensor b,
                  torch::Tensor c,
                  torch::Tensor batch_sizes);

}  // namespace disagmoe
