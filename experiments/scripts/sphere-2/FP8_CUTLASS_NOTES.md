# FP8 Grouped GEMM on SM89 (L40S) — Implementation Notes

## Problem

The original code used CUTLASS v3.2 which had no SM89 or FP8 support in the
2.x grouped GEMM API. The `MakeGemmGroupedFP8` template declared BF16×FP8
mixed-input with `Sm80` arch, but CUTLASS lacked the necessary template
specializations (`MmaTensorOpMultiplicandTileIterator` for FP8 types).

## Solution

### 1. CUTLASS Upgrade: v3.2.0 → v4.4.0

Intermediate versions were tried and rejected:
- **v3.5.0**: Added `Sm89` arch tag + `mma_sm89.h` FP8 MMA instructions, but
  the warp-level tile iterators still had zero FP8 specializations.
- **v4.4.0**: First version with complete FP8 tile iterator support in the 2.x
  GEMM pipeline (needed for `DefaultGemmGrouped`).

### 2. Native SM89 FP8×FP8 GEMM (`grouped_gemm.cu`)

Changed `MakeGemmGroupedFP8` from mixed BF16×FP8 to native FP8×FP8:

| Parameter        | Before (broken)       | After (working)            |
|------------------|-----------------------|----------------------------|
| ElementA         | `bfloat16_t`          | `float_e4m3_t`             |
| ElementB         | `float_e4m3_t`        | `float_e4m3_t`             |
| LayoutB          | `RowMajor`            | `ColumnMajor`              |
| ArchTag          | `Sm80`                | `Sm89`                     |
| InstructionShape | `16×8×16`             | `16×8×32`                  |
| AlignmentA       | 8                     | 16                         |

**Why ColumnMajor B?** CUTLASS FP8 tile iterators (`MmaTensorOpMultiplicandTileIterator`)
only have specializations for the canonical TN layout (A=RowMajor, B=ColumnMajor).
Weights are transposed `[E,K,N] → [E,N,K]` at init time.

### 3. Python-side FP8 Activation Quantization (`experts.py`)

Activations are quantized BF16→FP8 on the Python side using
`sglang_per_token_group_quant_fp8()` before calling `setup_meta()`,
following the same pattern as `MoEExpertsDeepGemmFP8`.

The C++ `bf16_to_fp8_kernel` was removed — `setup_meta` now asserts
input is already `Float8_e4m3fn`.

**group_size caveat**: Some models (e.g. gptoss_120b, hidden=2880) have
dimensions not divisible by 128. We fall back to `group_size=64` when
`dim % 128 != 0`.

### 4. CUTLASS v4.4.0 API Change

`GemmGrouped::Arguments` constructor changed to non-const pointer params.
Updated `reinterpret_cast<ElemA const * const *>` → `reinterpret_cast<ElemA**>`.

## Files Changed

- `third_party/cutlass` — submodule updated to v4.4.0
- `csrc/cuda_ops/grouped_gemm.cu` — FP8 GEMM template + runtime changes
- `csrc/include/grouped_gemm.h` — added `fp8_weight_T_` member
- `disagmoe/models/experts.py` — Python-side FP8 quantization

## Test Run Script

See `experiments/scripts/test_fp8_2gpu.sh` in this repo.
