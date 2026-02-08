#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <sys/stat.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/packed_stride.hpp"

#include "config.hpp"

/**
 * MoE Expert GEMM Profiler
 * =========================
 * 
 * Purpose: Profile per-expert GEMM latency for MoE models across batch sizes 1-512.
 * Compares plain GEMM (with CUDA graphs) vs grouped GEMM to measure kernel launch
 * overhead and batching benefits.
 * 
 * Kernels Used:
 * - Plain GEMM: cutlass::gemm::device::Gemm (SM80/SM90 tensor core optimized)
 *   Runs 8 sequential expert GEMMs captured in a CUDA graph to amortize launch overhead.
 * - Grouped GEMM: cutlass::gemm::device::GemmGrouped (SM80/SM90)
 *   Runs all 8 expert GEMMs in a single kernel launch using pointer arrays.
 * 
 * L2 Cache Busting via Workspace Rotation:
 * ----------------------------------------
 * Problem: In production, MoE layers run interleaved with attention and other ops,
 * so weight matrices are constantly evicted from L2. But microbenchmarking a single
 * GEMM repeatedly causes "L2 camping" - weights stay hot in cache, yielding
 * artificially low latencies that don't reflect real-world performance.
 * 
 * Solution: We allocate N complete copies of all tensors (A, B, C matrices), where
 * N is chosen such that the *total working set* across all workspaces comfortably
 * exceeds the device's L2 cache capacity (we target ~3x L2). Each workspace contains
 * independently allocated and randomly initialized tensors.
 * 
 * During profiling, we rotate through these workspace copies by changing the base
 * pointers for A/B/C used by each GEMM launch:
 *   - Iteration 0: use workspace[0] tensors
 *   - Iteration 1: use workspace[1] tensors
 *   - ...
 *   - Iteration N: use workspace[0] again (but now evicted from L2)
 *
 * Practically: we are "computing another set of experts" in the sense that we
 * run the same expert GEMMs but on another set of randomly-initialized weights and
 * activations of identical shape. Outputs are not consumed; this is purely to drive
 * realistic memory traffic during timing.
 *
 * For plain GEMM, the rotation sequence is captured into a CUDA graph so each graph
 * replay touches every workspace and preserves the intended cache-thrashing pattern.
 * 
 * Supported GPUs:
 * - SM80: NVIDIA A100 (compile with -arch=sm_80)
 * - SM90: NVIDIA H100/H200 (compile with -arch=sm_90a recommended for best performance)
 * 
 * Workflow:
 * 1. Detect GPU and select appropriate kernels
 * 2. For each model config (hidden_size, moe_intermediate_size from config.hpp)
 * 3. Allocate N workspace sets, each with random non-zero tensors
 * 4. For each per-expert batch_size in [1..512]:
 *    a. Profile plain GEMM: capture 8 expert GEMMs into graph, rotate workspaces
 *    b. Profile grouped GEMM: single kernel with 8 problems, rotate workspaces
 *    c. Compute per-expert time = total_time / num_experts
 * 5. Output CSV with (batch_size, avg_time_ms) for simulator consumption
 */

using namespace moe_gemm_profiler;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUTLASS_CHECK(status) do { \
    cutlass::Status s = status; \
    if (s != cutlass::Status::kSuccess) { \
        std::cerr << "CUTLASS Error: " << cutlass::cutlassGetStatusString(s) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// ============================================================================
// SM80 (A100) Kernel Definitions
// These also work on SM90 via forward compatibility, but are not Hopper-optimized.
// ============================================================================

using GemmConfigSm80 = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ElementA, ElementB, ElementC, ElementAccumulator>;

using PlainGemmSm80 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

using GroupedGemmKernelSm80 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, GemmConfigSm80::kAlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, GemmConfigSm80::kAlignmentB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    GemmConfigSm80::ThreadblockShape,
    GemmConfigSm80::WarpShape,
    GemmConfigSm80::InstructionShape,
    GemmConfigSm80::EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    GemmConfigSm80::kStages
>::GemmKernel;

using GroupedGemmSm80 = cutlass::gemm::device::GemmGrouped<GroupedGemmKernelSm80>;

// ============================================================================
// SM90 (H100/H200) Kernel Definitions (CUTLASS 3.x)
// ============================================================================

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace sm90 {

using ElementCompute = float;
using ElementScalar = float;

// 16B alignment enables TMA paths when available.
static constexpr int AlignmentA = 16 / sizeof(ElementA);
static constexpr int AlignmentB = 16 / sizeof(ElementB);
static constexpr int AlignmentC = 16 / sizeof(ElementC);
static constexpr int AlignmentD = 16 / sizeof(ElementC);

using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementC,
    ElementCompute,
    ElementScalar,
    RoundStyle>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementC,
    LayoutC,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

} // namespace sm90

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

struct GemmWorkspace {
    ElementA* d_A;
    ElementB* d_B;
    ElementC* d_C;
    size_t size_A;
    size_t size_B;
    size_t size_C;
};

struct ProfileResult {
    int batch_size;
    double plain_gemm_time_ms;
    double grouped_gemm_time_ms;
    double plain_per_expert_ms;
    double grouped_per_expert_ms;
};

// Forward declarations (helpers are defined before some implementations below).
void allocate_workspace(GemmWorkspace& ws, int M, int K, int N, int num_experts);
void free_workspace(GemmWorkspace& ws);
void initialize_tensor(void* ptr, size_t bytes, int workspace_id);

double profile_plain_gemm_with_cudagraph(
    const std::vector<GemmWorkspace>& workspaces,
    int M, int K, int N,
    int num_experts,
    int warmup_iters,
    int benchmark_iters,
    int max_m_per_expert,
    bool use_sm90);

double profile_grouped_gemm(
    const std::vector<GemmWorkspace>& workspaces,
    int M, int K, int N,
    int num_experts,
    int warmup_iters,
    int benchmark_iters,
    int max_m_per_expert,
    bool use_sm90);

static std::vector<GemmWorkspace> allocate_and_init_workspaces(
    int ws_count,
    int max_m_per_expert,
    int K,
    int N,
    int num_experts) {
    std::vector<GemmWorkspace> workspaces(ws_count);
    for (int i = 0; i < ws_count; ++i) {
        allocate_workspace(workspaces[i], max_m_per_expert, K, N, num_experts);
        initialize_tensor(workspaces[i].d_A, workspaces[i].size_A, i);
        initialize_tensor(workspaces[i].d_B, workspaces[i].size_B, i);
        initialize_tensor(workspaces[i].d_C, workspaces[i].size_C, i);
    }
    return workspaces;
}

static void free_workspaces(std::vector<GemmWorkspace>& workspaces) {
    for (auto& ws : workspaces) {
        free_workspace(ws);
    }
    workspaces.clear();
}

static std::vector<ProfileResult> run_one_profile_mode(
    std::vector<GemmWorkspace> const& workspaces,
    int K,
    int N,
    int num_experts,
    int max_m_per_expert,
    bool use_sm90) {
    std::vector<ProfileResult> results;
    results.reserve((profiling_config::BATCH_SIZE_MAX - profiling_config::BATCH_SIZE_MIN + 1) / profiling_config::BATCH_SIZE_STEP);

    std::cout << "Profiling batch sizes " << profiling_config::BATCH_SIZE_MIN
              << " to " << profiling_config::BATCH_SIZE_MAX << "...\n";

    for (int batch_size = profiling_config::BATCH_SIZE_MIN;
         batch_size <= profiling_config::BATCH_SIZE_MAX;
         batch_size += profiling_config::BATCH_SIZE_STEP) {

        if (batch_size % 64 == 0 || batch_size == 1) {
            std::cout << "  Per-expert batch size: " << batch_size << "..." << std::flush;
        }

        double plain_time = profile_plain_gemm_with_cudagraph(
            workspaces, batch_size, K, N, num_experts,
            profiling_config::WARMUP_ITERATIONS,
            profiling_config::BENCHMARK_ITERATIONS,
            max_m_per_expert,
            use_sm90);

        double grouped_time = profile_grouped_gemm(
            workspaces, batch_size, K, N, num_experts,
            profiling_config::WARMUP_ITERATIONS,
            profiling_config::BENCHMARK_ITERATIONS,
            max_m_per_expert,
            use_sm90);

        ProfileResult result;
        result.batch_size = batch_size;
        result.plain_gemm_time_ms = plain_time;
        result.grouped_gemm_time_ms = grouped_time;
        result.plain_per_expert_ms = plain_time / num_experts;
        result.grouped_per_expert_ms = grouped_time / num_experts;
        results.push_back(result);

        if (batch_size % 64 == 0 || batch_size == 1) {
            std::cout << " plain=" << std::fixed << std::setprecision(4) << result.plain_per_expert_ms
                      << "ms, grouped=" << result.grouped_per_expert_ms << "ms\n";
        }
    }

    return results;
}

int calculate_workspace_count(int64_t total_bytes, int64_t l2_cache_size) {
    if (profiling_config::WORKSPACE_COUNT > 0) {
        return profiling_config::WORKSPACE_COUNT;
    }
    int count = std::max(1, static_cast<int>((3 * l2_cache_size) / std::max(total_bytes, int64_t(1))) + 1);
    return std::min(count, 32);
}

void allocate_workspace(GemmWorkspace& ws, int M, int K, int N, int num_experts) {
    ws.size_A = static_cast<size_t>(num_experts) * static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(ElementA);
    ws.size_B = static_cast<size_t>(num_experts) * K * N * sizeof(ElementB);
    ws.size_C = static_cast<size_t>(num_experts) * static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(ElementC);
    
    CUDA_CHECK(cudaMalloc(&ws.d_A, ws.size_A));
    CUDA_CHECK(cudaMalloc(&ws.d_B, ws.size_B));
    CUDA_CHECK(cudaMalloc(&ws.d_C, ws.size_C));
}

void free_workspace(GemmWorkspace& ws) {
    if (ws.d_A) CUDA_CHECK(cudaFree(ws.d_A));
    if (ws.d_B) CUDA_CHECK(cudaFree(ws.d_B));
    if (ws.d_C) CUDA_CHECK(cudaFree(ws.d_C));
    ws.d_A = ws.d_B = ws.d_C = nullptr;
}

void initialize_tensor(void* ptr, size_t bytes, int workspace_id) {
    std::vector<uint8_t> host_data(bytes);
    std::mt19937 gen(42 + workspace_id * 12345);
    std::uniform_int_distribution<uint8_t> dist(1, 255);
    for (auto& b : host_data) b = dist(gen);
    CUDA_CHECK(cudaMemcpy(ptr, host_data.data(), bytes, cudaMemcpyHostToDevice));
}

double profile_plain_gemm_with_cudagraph(
    const std::vector<GemmWorkspace>& workspaces,
    int M, int K, int N,
    int num_experts,
    int warmup_iters,
    int benchmark_iters
    ,int max_m_per_expert
    ,bool use_sm90
) {
    int ws_count = static_cast<int>(workspaces.size());
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Plain GEMM: sequential experts, captured in a CUDA graph.

    if (!use_sm90) {
        PlainGemmSm80 gemm_op;

        auto make_args = [&](ElementA* a_ptr, ElementB* b_ptr, ElementC* c_ptr) {
            return typename PlainGemmSm80::Arguments(
                {M, N, K},
                {a_ptr, K},
                {b_ptr, K},
                {c_ptr, N},
                {c_ptr, N},
                {1.0f, 0.0f});
        };

        // Allocate a single workspace buffer (if required) for this GEMM size.
        void* d_workspace = nullptr;
        {
            auto const& ws0 = workspaces[0];
            auto args0 = make_args(ws0.d_A, ws0.d_B, ws0.d_C);
            size_t workspace_size = PlainGemmSm80::get_workspace_size(args0);
            if (workspace_size > 0) {
                CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
            }
        }

        for (int w = 0; w < warmup_iters; ++w) {
            auto const& ws = workspaces[w % ws_count];
            for (int e = 0; e < num_experts; ++e) {
                ElementA* a_ptr = ws.d_A + static_cast<size_t>(e) * max_m_per_expert * K;
                ElementB* b_ptr = ws.d_B + static_cast<size_t>(e) * K * N;
                ElementC* c_ptr = ws.d_C + static_cast<size_t>(e) * max_m_per_expert * N;

                auto args = make_args(a_ptr, b_ptr, c_ptr);
                CUTLASS_CHECK(gemm_op.initialize(args, d_workspace, stream));
                CUTLASS_CHECK(gemm_op(stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        int graph_iterations = 2 * ws_count;
        for (int iter = 0; iter < graph_iterations; ++iter) {
            auto const& ws = workspaces[iter % ws_count];
            for (int e = 0; e < num_experts; ++e) {
                ElementA* a_ptr = ws.d_A + static_cast<size_t>(e) * max_m_per_expert * K;
                ElementB* b_ptr = ws.d_B + static_cast<size_t>(e) * K * N;
                ElementC* c_ptr = ws.d_C + static_cast<size_t>(e) * max_m_per_expert * N;

                auto args = make_args(a_ptr, b_ptr, c_ptr);
                CUTLASS_CHECK(gemm_op.initialize(args, d_workspace, stream));
                CUTLASS_CHECK(gemm_op(stream));
            }
        }

        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < benchmark_iters; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

        double avg_time_ms = total_ms / (benchmark_iters * graph_iterations);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return avg_time_ms;
    }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    {
        using GemmSm90 = sm90::Gemm;

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

        GemmSm90 gemm_op;

        using StrideA = typename GemmSm90::GemmKernel::StrideA;
        using StrideB = typename GemmSm90::GemmKernel::StrideB;
        using StrideC = typename GemmSm90::GemmKernel::StrideC;
        using StrideD = typename GemmSm90::GemmKernel::StrideD;

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(max_m_per_expert, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(max_m_per_expert, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(max_m_per_expert, N, 1));

        auto make_args = [&](ElementA* a_ptr, ElementB* b_ptr, ElementC* c_ptr) {
            typename GemmSm90::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M, N, K, 1},
                {a_ptr, stride_A, b_ptr, stride_B},
                {{}, c_ptr, stride_C, c_ptr, stride_D},
                hw_info};
            args.epilogue.thread.alpha = 1.0f;
            args.epilogue.thread.beta = 0.0f;
            return args;
        };

        void* d_workspace = nullptr;
        {
            auto const& ws0 = workspaces[0];
            auto args0 = make_args(ws0.d_A, ws0.d_B, ws0.d_C);
            size_t workspace_size = GemmSm90::get_workspace_size(args0);
            if (workspace_size > 0) {
                CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
            }
            CUTLASS_CHECK(gemm_op.initialize(args0, d_workspace, stream));
        }

        for (int w = 0; w < warmup_iters; ++w) {
            auto const& ws = workspaces[w % ws_count];
            for (int e = 0; e < num_experts; ++e) {
                ElementA* a_ptr = ws.d_A + static_cast<size_t>(e) * max_m_per_expert * K;
                ElementB* b_ptr = ws.d_B + static_cast<size_t>(e) * K * N;
                ElementC* c_ptr = ws.d_C + static_cast<size_t>(e) * max_m_per_expert * N;
                auto args = make_args(a_ptr, b_ptr, c_ptr);
                CUTLASS_CHECK(gemm_op.update(args, d_workspace));
                CUTLASS_CHECK(gemm_op.run(stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        int graph_iterations = 2 * ws_count;
        for (int iter = 0; iter < graph_iterations; ++iter) {
            auto const& ws = workspaces[iter % ws_count];
            for (int e = 0; e < num_experts; ++e) {
                ElementA* a_ptr = ws.d_A + static_cast<size_t>(e) * max_m_per_expert * K;
                ElementB* b_ptr = ws.d_B + static_cast<size_t>(e) * K * N;
                ElementC* c_ptr = ws.d_C + static_cast<size_t>(e) * max_m_per_expert * N;
                auto args = make_args(a_ptr, b_ptr, c_ptr);
                CUTLASS_CHECK(gemm_op.update(args, d_workspace));
                CUTLASS_CHECK(gemm_op.run(stream));
            }
        }

        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < benchmark_iters; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

        double avg_time_ms = total_ms / (benchmark_iters * graph_iterations);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return avg_time_ms;
    }
#else
    (void)max_m_per_expert;
    (void)use_sm90;
    std::cerr << "SM90 requested but CUTLASS_ARCH_MMA_SM90_SUPPORTED is not defined at compile time." << std::endl;
    exit(1);
#endif
}

double profile_grouped_gemm(
    const std::vector<GemmWorkspace>& workspaces,
    int M, int K, int N,
    int num_experts,
    int warmup_iters,
    int benchmark_iters
    ,int max_m_per_expert
    ,bool use_sm90
) {
    int ws_count = static_cast<int>(workspaces.size());
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    if (!use_sm90) {
        std::vector<cutlass::gemm::GemmCoord> problem_sizes(num_experts, {M, N, K});
        std::vector<int64_t> lda(num_experts, K);
        std::vector<int64_t> ldb(num_experts, K);
        std::vector<int64_t> ldc(num_experts, N);

        std::vector<ElementA*> ptr_A(num_experts);
        std::vector<ElementB*> ptr_B(num_experts);
        std::vector<ElementC*> ptr_C(num_experts);

        cutlass::gemm::GemmCoord* d_problem_sizes;
        int64_t* d_lda, *d_ldb, *d_ldc;
        ElementA** d_ptr_A;
        ElementB** d_ptr_B;
        ElementC** d_ptr_C;

        CUDA_CHECK(cudaMalloc(&d_problem_sizes, num_experts * sizeof(cutlass::gemm::GemmCoord)));
        CUDA_CHECK(cudaMalloc(&d_lda, num_experts * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_ldb, num_experts * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_ldc, num_experts * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_ptr_A, num_experts * sizeof(ElementA*)));
        CUDA_CHECK(cudaMalloc(&d_ptr_B, num_experts * sizeof(ElementB*)));
        CUDA_CHECK(cudaMalloc(&d_ptr_C, num_experts * sizeof(ElementC*)));

        CUDA_CHECK(cudaMemcpy(d_problem_sizes, problem_sizes.data(), num_experts * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lda, lda.data(), num_experts * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ldb, ldb.data(), num_experts * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ldc, ldc.data(), num_experts * sizeof(int64_t), cudaMemcpyHostToDevice));

        GroupedGemmSm80 grouped_gemm;
        typename GroupedGemmSm80::EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);
    
        for (int w = 0; w < warmup_iters; ++w) {
            auto const& ws = workspaces[w % ws_count];
            for (int e = 0; e < num_experts; ++e) {
                ptr_A[e] = ws.d_A + static_cast<size_t>(e) * max_m_per_expert * K;
                ptr_B[e] = ws.d_B + static_cast<size_t>(e) * K * N;
                ptr_C[e] = ws.d_C + static_cast<size_t>(e) * max_m_per_expert * N;
            }
            CUDA_CHECK(cudaMemcpy(d_ptr_A, ptr_A.data(), num_experts * sizeof(ElementA*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ptr_B, ptr_B.data(), num_experts * sizeof(ElementB*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ptr_C, ptr_C.data(), num_experts * sizeof(ElementC*), cudaMemcpyHostToDevice));

            int threadblock_count = grouped_gemm.sufficient(problem_sizes.data(), num_experts);

            typename GroupedGemmSm80::Arguments args(
                d_problem_sizes,
                num_experts,
                threadblock_count,
                epilogue_params,
                d_ptr_A, d_ptr_B, d_ptr_C, d_ptr_C,
                d_lda, d_ldb, d_ldc, d_ldc,
                nullptr);

            size_t workspace_size = grouped_gemm.get_workspace_size(args);
            void* d_workspace = nullptr;
            if (workspace_size > 0) {
                CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
            }

            CUTLASS_CHECK(grouped_gemm.initialize(args, d_workspace));
            CUTLASS_CHECK(grouped_gemm.run(stream));

            if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    void* d_workspace = nullptr;
    size_t max_workspace_size = 0;
    
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iter = 0; iter < benchmark_iters; ++iter) {
            auto const& ws = workspaces[iter % ws_count];
            for (int e = 0; e < num_experts; ++e) {
                ptr_A[e] = ws.d_A + static_cast<size_t>(e) * max_m_per_expert * K;
                ptr_B[e] = ws.d_B + static_cast<size_t>(e) * K * N;
                ptr_C[e] = ws.d_C + static_cast<size_t>(e) * max_m_per_expert * N;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_ptr_A, ptr_A.data(), num_experts * sizeof(ElementA*), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_ptr_B, ptr_B.data(), num_experts * sizeof(ElementB*), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_ptr_C, ptr_C.data(), num_experts * sizeof(ElementC*), cudaMemcpyHostToDevice, stream));

            int threadblock_count = grouped_gemm.sufficient(problem_sizes.data(), num_experts);

            typename GroupedGemmSm80::Arguments args(
                d_problem_sizes,
                num_experts,
                threadblock_count,
                epilogue_params,
                d_ptr_A, d_ptr_B, d_ptr_C, d_ptr_C,
                d_lda, d_ldb, d_ldc, d_ldc,
                nullptr);

            size_t workspace_size = grouped_gemm.get_workspace_size(args);
            if (workspace_size > max_workspace_size) {
                if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
                CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
                max_workspace_size = workspace_size;
            }

            CUTLASS_CHECK(grouped_gemm.initialize(args, d_workspace));
            CUTLASS_CHECK(grouped_gemm.run(stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    
    double avg_time_ms = total_ms / benchmark_iters;
    
        if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
        CUDA_CHECK(cudaFree(d_problem_sizes));
        CUDA_CHECK(cudaFree(d_lda));
        CUDA_CHECK(cudaFree(d_ldb));
        CUDA_CHECK(cudaFree(d_ldc));
        CUDA_CHECK(cudaFree(d_ptr_A));
        CUDA_CHECK(cudaFree(d_ptr_B));
        CUDA_CHECK(cudaFree(d_ptr_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return avg_time_ms;
    }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    {
        // Hopper path: use a single strided-batched GEMM with L = num_experts.
        using GemmSm90 = sm90::Gemm;

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

        GemmSm90 gemm_op;

        using StrideA = typename GemmSm90::GemmKernel::StrideA;
        using StrideB = typename GemmSm90::GemmKernel::StrideB;
        using StrideC = typename GemmSm90::GemmKernel::StrideC;
        using StrideD = typename GemmSm90::GemmKernel::StrideD;

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(max_m_per_expert, K, num_experts));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, num_experts));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(max_m_per_expert, N, num_experts));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(max_m_per_expert, N, num_experts));

        auto make_args = [&](ElementA* a_ptr, ElementB* b_ptr, ElementC* c_ptr) {
            typename GemmSm90::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M, N, K, num_experts},
                {a_ptr, stride_A, b_ptr, stride_B},
                {{}, c_ptr, stride_C, c_ptr, stride_D},
                hw_info};
            args.epilogue.thread.alpha = 1.0f;
            args.epilogue.thread.beta = 0.0f;
            return args;
        };

        void* d_workspace = nullptr;
        {
            auto const& ws0 = workspaces[0];
            auto args0 = make_args(ws0.d_A, ws0.d_B, ws0.d_C);
            size_t workspace_size = GemmSm90::get_workspace_size(args0);
            if (workspace_size > 0) {
                CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
            }
            CUTLASS_CHECK(gemm_op.initialize(args0, d_workspace, stream));
        }

        for (int w = 0; w < warmup_iters; ++w) {
            auto const& ws = workspaces[w % ws_count];
            auto args = make_args(ws.d_A, ws.d_B, ws.d_C);
            CUTLASS_CHECK(gemm_op.update(args, d_workspace));
            CUTLASS_CHECK(gemm_op.run(stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iter = 0; iter < benchmark_iters; ++iter) {
            auto const& ws = workspaces[iter % ws_count];
            auto args = make_args(ws.d_A, ws.d_B, ws.d_C);
            CUTLASS_CHECK(gemm_op.update(args, d_workspace));
            CUTLASS_CHECK(gemm_op.run(stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

        double avg_time_ms = total_ms / benchmark_iters;

        if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return avg_time_ms;
    }
#else
    (void)max_m_per_expert;
    (void)use_sm90;
    std::cerr << "SM90 requested but CUTLASS_ARCH_MMA_SM90_SUPPORTED is not defined at compile time." << std::endl;
    exit(1);
#endif
}

void write_csv(const std::string& filename, const std::vector<ProfileResult>& results) {
    std::ofstream file(filename);
    file << "batch_size,avg_time_ms\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& r : results) {
        file << r.batch_size << "," << r.grouped_per_expert_ms << "\n";
    }
    file.close();
    std::cout << "Wrote CSV: " << filename << std::endl;
}

void write_detailed_csv(const std::string& filename, const std::vector<ProfileResult>& results) {
    std::ofstream file(filename);
    file << "batch_size,plain_gemm_total_ms,grouped_gemm_total_ms,plain_per_expert_ms,grouped_per_expert_ms\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& r : results) {
        file << r.batch_size << "," 
             << r.plain_gemm_time_ms << ","
             << r.grouped_gemm_time_ms << ","
             << r.plain_per_expert_ms << ","
             << r.grouped_per_expert_ms << "\n";
    }
    file.close();
    std::cout << "Wrote detailed CSV: " << filename << std::endl;
}

void write_summary(const std::string& filename, const MoELayerConfig& model, 
                   const std::string& gpu_name, const std::vector<ProfileResult>& results) {
    std::ofstream file(filename);
    
    file << "========================================\n";
    file << "MoE GEMM Profiling Summary\n";
    file << "========================================\n\n";
    
    file << "Model: " << model.model_name << "\n";
    file << "GPU: " << gpu_name << "\n";
    file << "Hidden Size: " << model.hidden_size << "\n";
    file << "MoE Intermediate Size: " << model.moe_intermediate_size << "\n";
    file << "Number of Experts: " << model.num_experts << "\n";
    file << "Experts per Token: " << model.num_experts_per_tok << "\n\n";
    
    file << "Profiling Configuration:\n";
    file << "  Warmup Iterations: " << profiling_config::WARMUP_ITERATIONS << "\n";
    file << "  Benchmark Iterations: " << profiling_config::BENCHMARK_ITERATIONS << "\n";
    file << "  Num Experts (per call): " << profiling_config::NUM_EXPERTS << "\n\n";
    
    double avg_speedup = 0.0;
    int count = 0;
    for (const auto& r : results) {
        if (r.plain_per_expert_ms > 0) {
            avg_speedup += r.plain_per_expert_ms / r.grouped_per_expert_ms;
            count++;
        }
    }
    avg_speedup /= count;
    
    file << "Results Summary:\n";
    file << "  Batch Size Range: " << results.front().batch_size << " - " << results.back().batch_size << "\n";
    file << "  Average GroupedGEMM Speedup over Plain GEMM: " << std::fixed << std::setprecision(2) << avg_speedup << "x\n\n";
    
    file << "Sample Results (batch_size: plain_ms -> grouped_ms):\n";
    std::vector<int> sample_sizes = {1, 16, 32, 64, 128, 256, 512};
    for (int bs : sample_sizes) {
        for (const auto& r : results) {
            if (r.batch_size == bs) {
                file << "  " << std::setw(3) << bs << ": " 
                     << std::fixed << std::setprecision(4) << r.plain_per_expert_ms 
                     << " ms -> " << r.grouped_per_expert_ms << " ms\n";
                break;
            }
        }
    }
    
    file.close();
    std::cout << "Wrote summary: " << filename << std::endl;
}

void run_profiler_for_model(const MoELayerConfig& model, const std::string& gpu_name) {
    std::cout << "\n========================================\n";
    std::cout << "Profiling: " << model.model_name << " on " << gpu_name << "\n";
    std::cout << "========================================\n";
    
    int K = model.hidden_size;
    int N = model.moe_intermediate_size;
    int num_experts = profiling_config::NUM_EXPERTS;
    
    int max_batch_size = profiling_config::BATCH_SIZE_MAX;
    int max_m_per_expert = max_batch_size;
    int64_t total_bytes_per_ws = static_cast<int64_t>(num_experts) * (
        static_cast<int64_t>(max_batch_size) * K * sizeof(ElementA) +
        static_cast<int64_t>(K) * N * sizeof(ElementB) +
        static_cast<int64_t>(max_batch_size) * N * sizeof(ElementC)
    );

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    int64_t l2_cache_size = static_cast<int64_t>(props.l2CacheSize);
    bool use_sm90 = (props.major == 9);

    // Two REAL modes:
    // - No rotation: ws_count = 1 (always reuse the same allocations)
    // - Rotation: ws_count = auto (or explicitly overridden via config)
    int ws_count_rotation = calculate_workspace_count(total_bytes_per_ws, l2_cache_size);
    int ws_count_no_rotation = 1;

    std::cout << "\n[Mode] No rotation (ws_count=" << ws_count_no_rotation << ")\n";
    auto workspaces_no_rot = allocate_and_init_workspaces(ws_count_no_rotation, max_m_per_expert, K, N, num_experts);
    auto results_no_rot = run_one_profile_mode(workspaces_no_rot, K, N, num_experts, max_m_per_expert, use_sm90);
    free_workspaces(workspaces_no_rot);

    std::cout << "\n[Mode] Workspace rotation (ws_count=" << ws_count_rotation << ")\n";
    auto workspaces_rot = allocate_and_init_workspaces(ws_count_rotation, max_m_per_expert, K, N, num_experts);
    auto results_rot = run_one_profile_mode(workspaces_rot, K, N, num_experts, max_m_per_expert, use_sm90);
    free_workspaces(workspaces_rot);
    
    mkdir(output_config::PLOTS_DIR, 0755);
    
    std::string csv_dir = std::string(output_config::CSV_OUTPUT_DIR);
    mkdir(csv_dir.c_str(), 0755);
    
    std::string base_name = std::string(model.model_name) + "_" + gpu_name;

    // Simulator consumption: keep default file as the rotation (cache-busted) measurement.
    write_csv(csv_dir + "/" + base_name + ".csv", results_rot);
    write_csv(csv_dir + "/" + base_name + "_no_rotation.csv", results_no_rot);

    write_detailed_csv(std::string(output_config::PLOTS_DIR) + "/" + base_name + "_rotated_detailed.csv", results_rot);
    write_detailed_csv(std::string(output_config::PLOTS_DIR) + "/" + base_name + "_no_rotation_detailed.csv", results_no_rot);
    write_summary(std::string(output_config::PLOTS_DIR) + "/" + base_name + "_summary.txt", model, gpu_name, results_rot);
}

int main() {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    
    std::cout << "========================================\n";
    std::cout << "MoE GEMM Profiler\n";
    std::cout << "========================================\n";
    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "L2 Cache Size: " << (props.l2CacheSize >> 20) << " MB\n";
    std::cout << "SM Count: " << props.multiProcessorCount << "\n";
    
    std::string gpu_name;
    if (props.major == 8 && props.minor == 0) {
        gpu_name = "A100";
        if (!gpu_config::ENABLE_A100) {
            std::cout << "A100 profiling disabled in config. Exiting.\n";
            return 0;
        }
    } else if (props.major == 9 && props.minor == 0) {
        gpu_name = "H200";
        if (!gpu_config::ENABLE_H200) {
            std::cout << "H200 profiling disabled in config. Exiting.\n";
            return 0;
        }
    } else {
        std::cout << "Warning: Unknown GPU architecture. Using 'SM" << props.major << props.minor << "'\n";
        gpu_name = "SM" + std::to_string(props.major) + std::to_string(props.minor);
    }
    
    auto models = get_all_models();
    for (const auto& model : models) {
        run_profiler_for_model(model, gpu_name);
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Profiling Complete!\n";
    std::cout << "========================================\n";
    std::cout << "Results written to:\n";
    std::cout << "  - " << output_config::CSV_OUTPUT_DIR << "/{model}_{gpu}.csv\n";
    std::cout << "  - " << output_config::PLOTS_DIR << "/{model}_{gpu}_detailed.csv\n";
    std::cout << "  - " << output_config::PLOTS_DIR << "/{model}_{gpu}_summary.txt\n";
    
    return 0;
}
