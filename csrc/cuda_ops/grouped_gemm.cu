/// Grouped GEMM wrapper for DisagMoE expert projections (sm < 90).
///
/// Two CUTLASS kernel instantiations are compiled:
///   LARGE  32x256x64  3 stages  -- for GPUs with >=164KB (A100 etc.)
///   SMALL  32x128x64  3 stages  -- for GPUs with <164KB  (L40S etc.)
///
/// init_grouped_gemm()          — probes hardware, selects tile config.
/// CutlassGemmRunner(w, max_t)  — per-weight-tensor: allocates buffers, calls initialize().
/// runner.setup_meta(a, c, bs)  — per-call metadata update (graph-capturable).
/// runner.run()                 — launches only the CUTLASS kernel.

#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include <memory>
#include <functional>

namespace disagmoe {

#define CUDA_CALL(code)                                               \
  do {                                                                \
    cudaError_t st = code;                                            \
    TORCH_CHECK(st == cudaSuccess, cudaGetErrorString(st));           \
  } while (0)

// Common epilogue for both configs
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ::cutlass::bfloat16_t, 8, float, float>;

// Helper: build a GemmGrouped type from tile parameters
template <int TbM, int TbN, int TbK, int WpM, int WpN, int WpK, int Stages>
using MakeGemmGrouped = ::cutlass::gemm::device::GemmGrouped<
    typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ::cutlass::bfloat16_t, ::cutlass::layout::RowMajor,
        ::cutlass::ComplexTransform::kNone, 8,
        ::cutlass::bfloat16_t, ::cutlass::layout::RowMajor,
        ::cutlass::ComplexTransform::kNone, 8,
        ::cutlass::bfloat16_t, ::cutlass::layout::RowMajor, float,
        ::cutlass::arch::OpClassTensorOp, ::cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TbM, TbN, TbK>,
        cutlass::gemm::GemmShape<WpM, WpN, WpK>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        EpilogueOp,
        ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Stages>::GemmKernel>;

// LARGE: 32x256x64, warp 32x64x64, 3 stages
// using GemmGroupedLarge = MakeGemmGrouped<128, 256, 64, 64, 64, 64, 3>; // for large bsz
using GemmGroupedLarge = MakeGemmGrouped<32, 256, 64, 32, 64, 64, 3>; // for usual bsz

// SMALL: 32x128x64, warp 32x64x64, 3 stages
// using GemmGroupedSmall = MakeGemmGrouped<128, 128, 64, 64, 64, 64, 3>; // for large bsz
using GemmGroupedSmall = MakeGemmGrouped<32, 128, 64, 32, 64, 64, 3>; // for usual bsz

// Cached state (set once by init_grouped_gemm)
enum class TileConfig : int { UNINITIALIZED = 0, LARGE = 1, SMALL = 2 };

static TileConfig s_tile_config = TileConfig::UNINITIALIZED;
static int        s_device_id   = -1;

using Element = ::cutlass::bfloat16_t;

// --------------------------------------------------------------------------
// GPU setup kernel (graph-capturable)
//
// Reads batch_sizes from device memory, computes prefix-sum offsets, and
// writes the CUTLASS argument arrays (problems, ptr_a, ptr_c).
// Launch config is static (<<<1,1>>>), making it graph-capturable.
// --------------------------------------------------------------------------

__global__ void setup_grouped_gemm_args_kernel(
    const int64_t* __restrict__ batch_sizes,  // [E]
    cutlass::gemm::GemmCoord* __restrict__ problems,  // [E]
    Element** __restrict__ ptr_a,             // [E]
    Element** __restrict__ ptr_c,             // [E]
    Element* a_base,
    Element* c_base,
    int K,
    int N,
    int num_experts
) {
    // E is small (typically 8-64), single-thread sequential scan is fine.
    if (threadIdx.x == 0) {
        int64_t offset = 0;
        for (int i = 0; i < num_experts; ++i) {
            int64_t M = batch_sizes[i];
            problems[i] = cutlass::gemm::GemmCoord(
                static_cast<int>(M), N, K);
            ptr_a[i] = a_base + offset * K;
            ptr_c[i] = c_base + offset * N;
            offset += M;
        }
    }
}

// --------------------------------------------------------------------------
// Helper: initialize CUTLASS for a runner
// --------------------------------------------------------------------------

template <typename GemmGroupedT>
static std::function<void(cudaStream_t)>
init_cutlass_for_runner(
    torch::Tensor& workspace_out,
    torch::Tensor d_problems,
    torch::Tensor d_ptr_a, torch::Tensor d_ptr_b, torch::Tensor d_ptr_c,
    torch::Tensor d_lda, torch::Tensor d_ldb, torch::Tensor d_ldc,
    int64_t num_experts, int64_t max_tokens_per_expert,
    int64_t K, int64_t N)
{
    // Build host-side problem list for sufficient() computation
    std::vector<cutlass::gemm::GemmCoord> host_problems(num_experts);
    for (int64_t i = 0; i < num_experts; ++i) {
        host_problems[i] = cutlass::gemm::GemmCoord(
            static_cast<int>(max_tokens_per_expert),
            static_cast<int>(N),
            static_cast<int>(K));
    }
    int threadblock_count = GemmGroupedT::sufficient(
        host_problems.data(), static_cast<int>(num_experts));
    TORCH_CHECK(threadblock_count > 0,
                "CUTLASS grouped GEMM: sufficient() returned 0.");

    // Write initial max-size problems to device for initialize()
    CUDA_CALL(cudaMemcpy(
        d_problems.data_ptr(), host_problems.data(),
        num_experts * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice));

    // Construct CUTLASS Arguments pointing to pre-allocated device buffers
    typename GemmGroupedT::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);
    typename GemmGroupedT::Arguments arguments(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(d_problems.data_ptr()),
        static_cast<int>(num_experts),
        threadblock_count,
        epilogue,
        reinterpret_cast<Element**>(d_ptr_a.data_ptr()),
        reinterpret_cast<Element**>(d_ptr_b.data_ptr()),
        reinterpret_cast<Element**>(d_ptr_c.data_ptr()),
        reinterpret_cast<Element**>(d_ptr_c.data_ptr()),  // D = C (in-place)
        d_lda.data_ptr<int64_t>(),
        d_ldb.data_ptr<int64_t>(),
        d_ldc.data_ptr<int64_t>(),
        d_ldc.data_ptr<int64_t>(),
        /*host_problem_sizes=*/nullptr);

    // Allocate CUTLASS workspace
    auto gemm = std::make_shared<GemmGroupedT>();
    int64_t ws_size = gemm->get_workspace_size(arguments);
    workspace_out = torch::empty(
        std::max(ws_size, int64_t(1)),
        torch::TensorOptions().dtype(torch::kInt8).device(d_problems.device()));

    // Initialize CUTLASS (sets internal params_ & smem config)
    auto status = gemm->initialize(arguments, workspace_out.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM initialize() failed (status ",
                static_cast<int>(status), ")");

    // Return type-erased kernel launcher
    return [gemm](cudaStream_t stream) {
        auto s = gemm->run(stream);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "CUTLASS grouped GEMM run() failed (status ",
                    static_cast<int>(s), ")");
    };
}

// --------------------------------------------------------------------------
// Public API: init_grouped_gemm
// --------------------------------------------------------------------------

std::string init_grouped_gemm(int device_id) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device_id));
    int sm = prop.major * 10 + prop.minor;
    TORCH_CHECK(sm < 90,
                "disagmoe_c.init_grouped_gemm: sm", sm,
                " is >= 90 -- use DeepGEMM instead.");

    int max_smem = 0;
    CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));

    constexpr int kLargeSmem = static_cast<int>(
        sizeof(typename GemmGroupedLarge::GemmKernel::SharedStorage));
    constexpr int kSmallSmem = static_cast<int>(
        sizeof(typename GemmGroupedSmall::GemmKernel::SharedStorage));

    std::string desc;
    if (max_smem >= kLargeSmem) {
        s_tile_config = TileConfig::LARGE;
        desc = "LARGE (32x256x64, 3 stages, " + std::to_string(kLargeSmem) +
               "B smem)";
    } else if (max_smem >= kSmallSmem) {
        s_tile_config = TileConfig::SMALL;
        desc = "SMALL (32x128x64, 3 stages, " + std::to_string(kSmallSmem) +
               "B smem)";
    } else {
        TORCH_CHECK(false,
                    "GPU shared memory (", max_smem,
                    "B) is too small for any CUTLASS grouped GEMM config. "
                    "Minimum required: ", kSmallSmem, "B.");
    }

    s_device_id = device_id;
    desc = "sm" + std::to_string(sm) + " / " + prop.name +
           " / max_smem=" + std::to_string(max_smem) +
           "B -> " + desc;
    return desc;
}

// --------------------------------------------------------------------------
// CutlassGemmRunner implementation
// --------------------------------------------------------------------------

CutlassGemmRunner::CutlassGemmRunner(torch::Tensor b_weight, int64_t max_tokens) {
    TORCH_CHECK(s_tile_config != TileConfig::UNINITIALIZED,
                "CutlassGemmRunner: call init_grouped_gemm() first.");
    TORCH_CHECK(b_weight.is_cuda() && b_weight.scalar_type() == torch::kBFloat16,
                "b_weight must be a CUDA bf16 tensor");
    TORCH_CHECK(b_weight.ndimension() == 3, "b_weight must be 3D [E, K, N]");

    int64_t E = b_weight.size(0);
    int64_t K = b_weight.size(1);
    int64_t N = b_weight.size(2);

    num_experts_ = E;
    K_ = K;
    N_ = N;
    b_weight_ = b_weight;  // prevent GC

    auto dev   = b_weight.device();
    auto o_i8  = torch::TensorOptions().dtype(torch::kInt8).device(dev);
    auto o_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    // Pre-allocate device argument arrays
    d_problems_ = torch::empty({static_cast<int64_t>(E * sizeof(cutlass::gemm::GemmCoord))}, o_i8);
    d_ptr_a_    = torch::empty({static_cast<int64_t>(E * sizeof(Element*))}, o_i8);
    d_ptr_b_    = torch::empty({static_cast<int64_t>(E * sizeof(Element*))}, o_i8);
    d_ptr_c_    = torch::empty({static_cast<int64_t>(E * sizeof(Element*))}, o_i8);
    d_lda_      = torch::empty({E}, o_i64);
    d_ldb_      = torch::empty({E}, o_i64);
    d_ldc_      = torch::empty({E}, o_i64);

    // Fill constant stride arrays
    d_lda_.fill_(K);
    d_ldb_.fill_(N);
    d_ldc_.fill_(N);

    // Fill weight pointers (constant, one-time H2D copy)
    std::vector<Element*> ptr_b_host(E);
    auto b_base = reinterpret_cast<Element*>(b_weight.data_ptr());
    for (int64_t i = 0; i < E; ++i) {
        ptr_b_host[i] = b_base + i * K * N;
    }
    CUDA_CALL(cudaMemcpy(d_ptr_b_.data_ptr(), ptr_b_host.data(),
                         E * sizeof(Element*), cudaMemcpyHostToDevice));

    // Initialize ptr_a and ptr_c with nullptrs (setup kernel will overwrite)
    CUDA_CALL(cudaMemset(d_ptr_a_.data_ptr(), 0, E * sizeof(Element*)));
    CUDA_CALL(cudaMemset(d_ptr_c_.data_ptr(), 0, E * sizeof(Element*)));

    // Compute max tokens per expert for threadblock sizing
    int64_t max_tpe = max_tokens / E;
    if (max_tpe <= 0) max_tpe = 1;

    // Dispatch to templated CUTLASS initialization
    switch (s_tile_config) {
        case TileConfig::LARGE:
            run_gemm_ = init_cutlass_for_runner<GemmGroupedLarge>(
                workspace_, d_problems_,
                d_ptr_a_, d_ptr_b_, d_ptr_c_,
                d_lda_, d_ldb_, d_ldc_,
                E, max_tpe, K, N);
            break;
        case TileConfig::SMALL:
            run_gemm_ = init_cutlass_for_runner<GemmGroupedSmall>(
                workspace_, d_problems_,
                d_ptr_a_, d_ptr_b_, d_ptr_c_,
                d_lda_, d_ldb_, d_ldc_,
                E, max_tpe, K, N);
            break;
        default:
            TORCH_CHECK(false, "unreachable");
    }

    CUDA_CALL(cudaDeviceSynchronize());
}

void CutlassGemmRunner::setup_meta(torch::Tensor a, torch::Tensor c,
                                    torch::Tensor batch_sizes) {
    TORCH_CHECK(a.is_cuda() && c.is_cuda(),
                "a, c must be CUDA tensors");
    TORCH_CHECK(batch_sizes.is_cuda() && batch_sizes.scalar_type() == torch::kInt64,
                "batch_sizes must be a CUDA int64 tensor");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    setup_grouped_gemm_args_kernel<<<1, 1, 0, stream>>>(
        batch_sizes.data_ptr<int64_t>(),
        reinterpret_cast<cutlass::gemm::GemmCoord*>(d_problems_.data_ptr()),
        reinterpret_cast<Element**>(d_ptr_a_.data_ptr()),
        reinterpret_cast<Element**>(d_ptr_c_.data_ptr()),
        reinterpret_cast<Element*>(a.data_ptr()),
        reinterpret_cast<Element*>(c.data_ptr()),
        static_cast<int>(K_),
        static_cast<int>(N_),
        static_cast<int>(num_experts_));
}

void CutlassGemmRunner::run() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    run_gemm_(stream);
}

}  // namespace disagmoe
