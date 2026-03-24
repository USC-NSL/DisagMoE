#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <functional>
#include <memory>

namespace disagmoe {

/// One-time initialization: detects hardware and selects optimal CUTLASS tile
/// configuration.  Must be called before creating any CutlassGemmRunner.
///
/// @param device_id   CUDA device ordinal
/// @return  Human-readable description of the selected config (for logging).
std::string init_grouped_gemm(int device_id);

/// Encapsulates a single CUTLASS grouped GEMM operation for one weight tensor.
///
/// Each instance owns its own device buffers and CUTLASS state.  No global
/// maps, no mutexes — each MoEExpertsCUTLASS module creates exactly two
/// runners (one for w13, one for w2) and calls them directly.
///
/// Usage:
///   runner = CutlassGemmRunner(weight, max_tokens)  // at init
///   runner.setup_meta(a, c, batch_sizes)            // per-call (graph-capturable)
///   runner.run()                                     // per-call (graph-capturable)
class CutlassGemmRunner {
public:
    /// Construct a runner for the given weight tensor.
    /// Allocates device buffers, calls CUTLASS initialize(), and pre-warms.
    ///
    /// @param b_weight    [num_experts, K, N] bf16 weight tensor (CUDA)
    /// @param max_tokens  Maximum total tokens across all experts
    CutlassGemmRunner(torch::Tensor b_weight, int64_t max_tokens);

    /// Update CUTLASS metadata arrays on device.
    ///
    /// Launches a small GPU kernel (<<<1,1>>>) that reads batch_sizes and
    /// writes the GemmCoord problems, ptr_a, and ptr_c arrays.
    /// Graph-capturable (static launch config, fixed buffer addresses).
    ///
    /// @param a            [total_tokens, K]     bf16 input  (CUDA)
    /// @param c            [total_tokens, N]     bf16 output (CUDA)
    /// @param batch_sizes  [num_experts]         int64 (CUDA)
    void setup_meta(torch::Tensor a, torch::Tensor c, torch::Tensor batch_sizes);

    /// Launch only the CUTLASS grouped GEMM kernel.
    ///
    /// setup_meta() MUST have been called first (either in the same graph
    /// capture or as a preceding operation) to populate the metadata arrays.
    /// Graph-capturable.
    void run();

private:
    torch::Tensor d_problems_;
    torch::Tensor d_ptr_a_, d_ptr_b_, d_ptr_c_;
    torch::Tensor d_lda_, d_ldb_, d_ldc_;
    torch::Tensor workspace_;
    torch::Tensor b_weight_;  // prevent GC

    int64_t num_experts_ = 0;
    int64_t K_ = 0;
    int64_t N_ = 0;

    std::function<void(cudaStream_t)> run_gemm_;
};

/// W8A16 variant: FP8 weights + per-channel float32 scales, dequantised to
/// BF16 internally via a fused CUDA kernel, then fed to a standard BF16
/// CutlassGemmRunner.
///
/// Usage:
///   runner = CutlassGemmRunnerFP8(fp8_weight, scale, max_tokens)
///   runner.setup_meta(a, c, batch_sizes)   // same API as BF16 runner
///   runner.run()
///   runner.dequant_weights()               // call after weight/scale update
class CutlassGemmRunnerFP8 {
public:
    /// @param fp8_weight     [E, K, N]  float8_e4m3fn weight tensor (CUDA)
    /// @param weight_scale   [E, N]     float32 per-channel scales  (CUDA)
    /// @param max_tokens     Maximum total tokens across all experts
    CutlassGemmRunnerFP8(torch::Tensor fp8_weight,
                          torch::Tensor weight_scale,
                          int64_t max_tokens);

    void setup_meta(torch::Tensor a, torch::Tensor c, torch::Tensor batch_sizes);

    /// Launches the CUTLASS W8A16 GEMM + post-GEMM scale correction.
    void run();

private:
    torch::Tensor fp8_weight_;     // [E, K, N] float8_e4m3fn — prevent GC
    torch::Tensor weight_scale_;   // [E, N]    float32       — prevent GC

    torch::Tensor d_problems_;
    torch::Tensor d_ptr_a_, d_ptr_b_, d_ptr_c_;
    torch::Tensor d_lda_, d_ldb_, d_ldc_;
    torch::Tensor workspace_;

    torch::Tensor batch_sizes_;    // cached from setup_meta for scale kernel
    torch::Tensor c_out_;          // cached from setup_meta for scale kernel
    torch::Tensor fp8_weight_T_;    // [E, N, K] transposed FP8 weights

    int64_t num_experts_ = 0;
    int64_t K_ = 0;
    int64_t N_ = 0;
    int64_t max_tokens_ = 0;

    std::function<void(cudaStream_t)> run_gemm_;
};

}  // namespace disagmoe
