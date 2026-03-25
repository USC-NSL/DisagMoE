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
#include "cutlass/float8.h"
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

// Helper: build a BF16×BF16 GemmGrouped type from tile parameters
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

// Helper: build a native SM89 FP8xFP8 GemmGrouped type.
// Uses SM89 FP8 tensor cores (mma.sync 16x8x32 e4m3).
// Activations are quantised BF16->FP8 online before the GEMM call.
template <int TbM, int TbN, int TbK, int WpM, int WpN, int WpK, int Stages>
using MakeGemmGroupedFP8 = ::cutlass::gemm::device::GemmGrouped<
    typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ::cutlass::float_e4m3_t, ::cutlass::layout::RowMajor,
        ::cutlass::ComplexTransform::kNone, 16,             // A: FP8, align 16
        ::cutlass::float_e4m3_t, ::cutlass::layout::ColumnMajor,
        ::cutlass::ComplexTransform::kNone, 16,             // B: FP8 col-major, align 16
        ::cutlass::bfloat16_t, ::cutlass::layout::RowMajor, float,
        ::cutlass::arch::OpClassTensorOp, ::cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<TbM, TbN, TbK>,
        cutlass::gemm::GemmShape<WpM, WpN, WpK>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp,
        ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Stages>::GemmKernel>;

// BF16 × BF16 kernel instantiations
using GemmGroupedLarge = MakeGemmGrouped<32, 256, 64, 32, 64, 64, 3>;
using GemmGroupedSmall = MakeGemmGrouped<32, 128, 64, 32, 64, 64, 3>;

// W8A16 (FP8 × BF16) kernel instantiations — same tile configs
using GemmGroupedFP8Large = MakeGemmGroupedFP8<32, 256, 64, 32, 64, 64, 3>;
using GemmGroupedFP8Small = MakeGemmGroupedFP8<32, 128, 64, 32, 64, 64, 3>;

// Cached state (set once by init_grouped_gemm)
enum class TileConfig : int { UNINITIALIZED = 0, LARGE = 1, SMALL = 2 };

static TileConfig s_tile_config = TileConfig::UNINITIALIZED;
static int        s_device_id   = -1;

using Element = ::cutlass::bfloat16_t;
using ElementFP8 = ::cutlass::float_e4m3_t;

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
// FP8 variant of the setup kernel — ptr_a holds ElementFP8 pointers.
// --------------------------------------------------------------------------

__global__ void setup_grouped_gemm_args_fp8_kernel(
    const int64_t* __restrict__ batch_sizes,
    cutlass::gemm::GemmCoord* __restrict__ problems,
    ElementFP8** __restrict__ ptr_a,
    Element**    __restrict__ ptr_c,
    ElementFP8* a_base,
    Element*    c_base,
    int K, int N, int num_experts
) {
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

    // Derive element types from the kernel (handles both BF16 and FP8 variants)
    using ElemA = typename GemmGroupedT::ElementA;
    using ElemB = typename GemmGroupedT::ElementB;
    using ElemC = typename GemmGroupedT::ElementC;

    // Construct CUTLASS Arguments pointing to pre-allocated device buffers
    typename GemmGroupedT::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);
    typename GemmGroupedT::Arguments arguments(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(d_problems.data_ptr()),
        static_cast<int>(num_experts),
        threadblock_count,
        epilogue,
        reinterpret_cast<ElemA**>(d_ptr_a.data_ptr()),
        reinterpret_cast<ElemB**>(d_ptr_b.data_ptr()),
        reinterpret_cast<ElemC**>(d_ptr_c.data_ptr()),
        reinterpret_cast<ElemC**>(d_ptr_c.data_ptr()),  // D = C
        reinterpret_cast<int64_t*>(d_lda.data_ptr()),
        reinterpret_cast<int64_t*>(d_ldb.data_ptr()),
        reinterpret_cast<int64_t*>(d_ldc.data_ptr()),
        reinterpret_cast<int64_t*>(d_ldc.data_ptr()),
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

// --------------------------------------------------------------------------
// Post-GEMM per-expert-per-channel scale correction for W8A16.
//
// The FP8 CUTLASS kernel computes C_raw = A_bf16 × B_fp8.
// The real result is C = C_raw × scale[expert, :] because the quantised
// weight satisfies  W_real = W_fp8 × scale.
//
// Tokens are packed by expert (expert 0 first, then 1, …).  The kernel
// determines expert assignment from batch_sizes via a short linear scan
// (E is typically 8–64, so this is cheap).
//
// Launch config: <<<ceil(max_tokens*N / 256), 256>>> — graph-capturable.
// --------------------------------------------------------------------------

__global__ void apply_w8a16_output_scale_kernel(
    const int64_t* __restrict__ batch_sizes,
    Element*       __restrict__ output,
    const float*   __restrict__ scale,
    int N, int num_experts
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int n     = static_cast<int>(idx % N);
    int64_t token = idx / N;

    int64_t offset = 0;
    for (int e = 0; e < num_experts; ++e) {
        int64_t next = offset + batch_sizes[e];
        if (token < next) {
            float val = static_cast<float>(output[idx]) * scale[e * N + n];
            output[idx] = Element(val);
            return;
        }
        offset = next;
    }
}

// --------------------------------------------------------------------------
// CutlassGemmRunnerFP8 implementation — native SM89 FP8 CUTLASS kernel
//
// Uses MakeGemmGroupedFP8 (both A and B are float_e4m3_t on SM89).
// BF16 activations are quantised to FP8 online before the GEMM.
// Post-GEMM per-channel scale correction restores the true output.
// --------------------------------------------------------------------------

CutlassGemmRunnerFP8::CutlassGemmRunnerFP8(
    torch::Tensor fp8_weight,
    torch::Tensor weight_scale,
    int64_t max_tokens)
{
    TORCH_CHECK(s_tile_config != TileConfig::UNINITIALIZED,
                "CutlassGemmRunnerFP8: call init_grouped_gemm() first.");
    TORCH_CHECK(fp8_weight.is_cuda() &&
                fp8_weight.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "fp8_weight must be a CUDA float8_e4m3fn tensor");
    TORCH_CHECK(fp8_weight.ndimension() == 3,
                "fp8_weight must be 3D [E, K, N]");
    TORCH_CHECK(weight_scale.is_cuda() &&
                weight_scale.scalar_type() == torch::kFloat32,
                "weight_scale must be a CUDA float32 tensor");
    TORCH_CHECK(weight_scale.ndimension() == 2,
                "weight_scale must be 2D [E, N]");
    TORCH_CHECK(fp8_weight.size(0) == weight_scale.size(0) &&
                fp8_weight.size(2) == weight_scale.size(1),
                "weight_scale shape [E, N] must match fp8_weight [E, K, N]");

    fp8_weight_   = fp8_weight;
    weight_scale_ = weight_scale;

    int64_t E = fp8_weight.size(0);
    int64_t K = fp8_weight.size(1);
    int64_t N = fp8_weight.size(2);

    num_experts_ = E;
    K_ = K;
    N_ = N;
    max_tokens_ = max_tokens;

    auto dev   = fp8_weight.device();
    auto o_i8  = torch::TensorOptions().dtype(torch::kInt8).device(dev);
    auto o_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    d_problems_ = torch::empty({static_cast<int64_t>(E * sizeof(cutlass::gemm::GemmCoord))}, o_i8);
    d_ptr_a_    = torch::empty({static_cast<int64_t>(E * sizeof(ElementFP8*))}, o_i8);
    d_ptr_b_    = torch::empty({static_cast<int64_t>(E * sizeof(ElementFP8*))}, o_i8);
    d_ptr_c_    = torch::empty({static_cast<int64_t>(E * sizeof(Element*))}, o_i8);
    d_lda_      = torch::empty({E}, o_i64);
    d_ldb_      = torch::empty({E}, o_i64);
    d_ldc_      = torch::empty({E}, o_i64);

    d_lda_.fill_(K);
    d_ldb_.fill_(K);   // ColumnMajor B: leading dim = K
    d_ldc_.fill_(N);

    // Transpose weights from [E, K, N] row-major to [E, N, K] row-major
    // (equivalent to [E, K, N] column-major — canonical TN layout for CUTLASS).
    fp8_weight_T_ = fp8_weight.transpose(1, 2).contiguous();

    // FP8 weight pointers into the transposed tensor
    std::vector<ElementFP8*> ptr_b_host(E);
    auto b_base = reinterpret_cast<ElementFP8*>(fp8_weight_T_.data_ptr());
    for (int64_t i = 0; i < E; ++i) {
        ptr_b_host[i] = b_base + i * N * K;
    }
    CUDA_CALL(cudaMemcpy(d_ptr_b_.data_ptr(), ptr_b_host.data(),
                         E * sizeof(ElementFP8*), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemset(d_ptr_a_.data_ptr(), 0, E * sizeof(ElementFP8*)));
    CUDA_CALL(cudaMemset(d_ptr_c_.data_ptr(), 0, E * sizeof(Element*)));

    int64_t max_tpe = max_tokens / E;
    if (max_tpe <= 0) max_tpe = 1;

    switch (s_tile_config) {
        case TileConfig::LARGE:
            run_gemm_ = init_cutlass_for_runner<GemmGroupedFP8Large>(
                workspace_, d_problems_,
                d_ptr_a_, d_ptr_b_, d_ptr_c_,
                d_lda_, d_ldb_, d_ldc_,
                E, max_tpe, K, N);
            break;
        case TileConfig::SMALL:
            run_gemm_ = init_cutlass_for_runner<GemmGroupedFP8Small>(
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

void CutlassGemmRunnerFP8::setup_meta(torch::Tensor a, torch::Tensor c,
                                       torch::Tensor batch_sizes) {
    TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "a must be a CUDA float8_e4m3fn tensor (quantize on Python side)");
    TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
    TORCH_CHECK(batch_sizes.is_cuda() && batch_sizes.scalar_type() == torch::kInt64,
                "batch_sizes must be a CUDA int64 tensor");

    batch_sizes_ = batch_sizes;
    c_out_       = c;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Activations are already FP8 -- set up CUTLASS argument arrays directly
    setup_grouped_gemm_args_fp8_kernel<<<1, 1, 0, stream>>>(
        batch_sizes.data_ptr<int64_t>(),
        reinterpret_cast<cutlass::gemm::GemmCoord*>(d_problems_.data_ptr()),
        reinterpret_cast<ElementFP8**>(d_ptr_a_.data_ptr()),
        reinterpret_cast<Element**>(d_ptr_c_.data_ptr()),
        reinterpret_cast<ElementFP8*>(a.data_ptr()),
        reinterpret_cast<Element*>(c.data_ptr()),
        static_cast<int>(K_),
        static_cast<int>(N_),
        static_cast<int>(num_experts_));
}

void CutlassGemmRunnerFP8::run() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    run_gemm_(stream);

    // Post-GEMM: apply per-expert-per-channel weight scale
    constexpr int kBlock = 256;
    int64_t max_elements = max_tokens_ * N_;
    int grid = static_cast<int>((max_elements + kBlock - 1) / kBlock);

    apply_w8a16_output_scale_kernel<<<grid, kBlock, 0, stream>>>(
        batch_sizes_.data_ptr<int64_t>(),
        reinterpret_cast<Element*>(c_out_.data_ptr()),
        weight_scale_.data_ptr<float>(),
        static_cast<int>(N_),
        static_cast<int>(num_experts_));
}

}  // namespace disagmoe
