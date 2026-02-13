/// Grouped GEMM wrapper for DisagMoE expert projections (sm < 90).
///
/// Two CUTLASS kernel instantiations are compiled:
///   LARGE  128x256x64  3 stages  144KB smem  -- for GPUs with >=164KB (A100 etc.)
///   SMALL  128x128x64  3 stages   96KB smem  -- for GPUs with <164KB  (L40S etc.)
///
/// init_grouped_gemm() probes the device once and selects the best config.
/// grouped_gemm() dispatches to the cached selection with zero runtime probing.

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

namespace disagmoe {

#define CUDA_CALL(code)                                               \
  do {                                                                \
    cudaError_t st = code;                                            \
    TORCH_CHECK(st == cudaSuccess, cudaGetErrorString(st));           \
  } while (0)

template <typename T>
static torch::Tensor CopyToDevice(const std::vector<T>& x,
                                  const torch::Device& device) {
    size_t bytes = x.size() * sizeof(T);
    auto out = torch::empty(
        static_cast<int64_t>(bytes),
        torch::TensorOptions().dtype(torch::kInt8).device(device));
    CUDA_CALL(cudaMemcpyAsync(out.data_ptr(), x.data(), bytes,
                              cudaMemcpyHostToDevice,
                              c10::cuda::getCurrentCUDAStream()));
    return out;
}

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

// LARGE: 128x256x64, warp 64x64x64, 3 stages -> ~144KB smem (A100-class, 164KB max optin)
using GemmGroupedLarge = MakeGemmGrouped<128, 256, 64, 64, 64, 64, 3>;

// SMALL: 128x128x64, warp 64x64x64, 3 stages -> ~96KB smem  (L40S-class, 99KB max optin)
using GemmGroupedSmall = MakeGemmGrouped<128, 128, 64, 64, 64, 64, 3>;

// Cached state (set once by init_grouped_gemm)
enum class TileConfig : int { UNINITIALIZED = 0, LARGE = 1, SMALL = 2 };

static TileConfig s_tile_config = TileConfig::UNINITIALIZED;
static int        s_device_id   = -1;

// Templated CUTLASS dispatch (shared by both configs)
template <typename GemmGroupedT>
static void CutlassGroupedGemmImpl(torch::Tensor a, torch::Tensor b,
                                    torch::Tensor c,
                                    const int64_t* batch_sizes_ptr,
                                    int64_t num_experts) {
    using Kernel   = typename GemmGroupedT::GemmKernel;
    using ElementA = ::cutlass::bfloat16_t;
    using ElementB = ::cutlass::bfloat16_t;
    using ElementC = ::cutlass::bfloat16_t;
    using LayoutA  = ::cutlass::layout::RowMajor;
    using LayoutB  = ::cutlass::layout::RowMajor;
    using LayoutC  = ::cutlass::layout::RowMajor;

    int64_t K = a.size(1);
    int64_t N = b.size(2);

    std::vector<cutlass::gemm::GemmCoord> problems(num_experts);
    std::vector<int64_t> lda(num_experts), ldb(num_experts), ldc(num_experts);
    std::vector<ElementA*> ptr_a(num_experts);
    std::vector<ElementB*> ptr_b(num_experts);
    std::vector<ElementC*> ptr_c(num_experts);

    int64_t offset_a = 0, offset_b = 0, offset_c = 0;
    for (int64_t i = 0; i < num_experts; ++i) {
        int64_t M = batch_sizes_ptr[i];
        problems[i] = cutlass::gemm::GemmCoord(
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
        lda[i] = LayoutA::packed({static_cast<int>(M), static_cast<int>(K)}).stride(0);
        ldb[i] = LayoutB::packed({static_cast<int>(K), static_cast<int>(N)}).stride(0);
        ldc[i] = LayoutC::packed({static_cast<int>(M), static_cast<int>(N)}).stride(0);
        ptr_a[i] = reinterpret_cast<ElementA*>(a.data_ptr()) + offset_a;
        ptr_b[i] = reinterpret_cast<ElementB*>(b.data_ptr()) + offset_b;
        ptr_c[i] = reinterpret_cast<ElementC*>(c.data_ptr()) + offset_c;
        offset_a += M * K;
        offset_b += b.size(1) * b.size(2);
        offset_c += M * N;
        if (M == 0) { problems[i].m() = 0; problems[i].n() = 0; }
    }

    auto dev = a.device();
    auto d_problems = CopyToDevice(problems, dev);
    auto d_ptr_a    = CopyToDevice(ptr_a, dev);
    auto d_ptr_b    = CopyToDevice(ptr_b, dev);
    auto d_ptr_c    = CopyToDevice(ptr_c, dev);
    auto d_lda      = CopyToDevice(lda, dev);
    auto d_ldb      = CopyToDevice(ldb, dev);
    auto d_ldc      = CopyToDevice(ldc, dev);

    int threadblock_count = GemmGroupedT::sufficient(
        problems.data(), static_cast<int>(num_experts));
    TORCH_CHECK(threadblock_count > 0,
                "CUTLASS grouped GEMM: sufficient() returned 0. "
                "SharedStorage=", int(sizeof(typename Kernel::SharedStorage)),
                " bytes -- this config does not fit the GPU.");

    typename GemmGroupedT::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);
    typename GemmGroupedT::Arguments arguments(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(d_problems.data_ptr()),
        static_cast<int>(num_experts), threadblock_count, epilogue,
        reinterpret_cast<ElementA**>(d_ptr_a.data_ptr()),
        reinterpret_cast<ElementB**>(d_ptr_b.data_ptr()),
        reinterpret_cast<ElementC**>(d_ptr_c.data_ptr()),
        reinterpret_cast<ElementC**>(d_ptr_c.data_ptr()),
        reinterpret_cast<int64_t*>(d_lda.data_ptr()),
        reinterpret_cast<int64_t*>(d_ldb.data_ptr()),
        reinterpret_cast<int64_t*>(d_ldc.data_ptr()),
        reinterpret_cast<int64_t*>(d_ldc.data_ptr()),
        /*host_problem_sizes=*/nullptr);

    GemmGroupedT gemm;
    int64_t ws_size = gemm.get_workspace_size(arguments);
    auto workspace = torch::empty(
        ws_size, torch::TensorOptions().dtype(torch::kInt8).device(dev));

    auto status = gemm.initialize(arguments, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM initialize() failed (status ",
                static_cast<int>(status), ")");

    status = gemm.run(c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM run() failed (status ",
                static_cast<int>(status), ")");
}

// APIs Exposed:

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
        desc = "LARGE (128x256x64, 3 stages, " + std::to_string(kLargeSmem) +
               "B smem)";
    } else if (max_smem >= kSmallSmem) {
        s_tile_config = TileConfig::SMALL;
        desc = "SMALL (128x128x64, 3 stages, " + std::to_string(kSmallSmem) +
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

void grouped_gemm(torch::Tensor a, torch::Tensor b,
                  torch::Tensor c, torch::Tensor batch_sizes) {
    TORCH_CHECK(s_tile_config != TileConfig::UNINITIALIZED,
                "disagmoe_c.grouped_gemm: call init_grouped_gemm() first.");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bf16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "b must be bf16");
    TORCH_CHECK(c.scalar_type() == torch::kBFloat16, "c must be bf16");
    TORCH_CHECK(a.ndimension() == 2, "a must be 2D [tokens, K]");
    TORCH_CHECK(b.ndimension() == 3, "b must be 3D [E, K, N]");
    TORCH_CHECK(c.ndimension() == 2, "c must be 2D [tokens, N]");
    TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64,
                "batch_sizes must be int64");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(batch_sizes.size(0) == b.size(0),
                "batch_sizes length must match b.size(0)");

    torch::Tensor bs_cpu = batch_sizes.is_cpu()
        ? batch_sizes : batch_sizes.to(torch::kCPU);
    const int64_t* bs_ptr = bs_cpu.data_ptr<int64_t>();
    int64_t num_experts = batch_sizes.size(0);

    switch (s_tile_config) {
        case TileConfig::LARGE:
            CutlassGroupedGemmImpl<GemmGroupedLarge>(a, b, c, bs_ptr, num_experts);
            break;
        case TileConfig::SMALL:
            CutlassGroupedGemmImpl<GemmGroupedSmall>(a, b, c, bs_ptr, num_experts);
            break;
        default:
            TORCH_CHECK(false, "unreachable");
    }
}

}  // namespace disagmoe
