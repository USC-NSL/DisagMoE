'''
Quantizer utility borrowed from sglang.
'''

import torch
from functools import lru_cache
from typing import Optional, Tuple

# ---- extracted helpers (from sglang.srt.utils.common) ----

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

def ceil_align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

HIP_FP8_E4M3_FNUZ_MAX = 224.0

def is_hip() -> bool:
    return torch.version.hip is not None

_is_hip = is_hip()

@lru_cache()
def is_fp8_fnuz() -> bool:
    if _is_hip:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
    return False

if is_fp8_fnuz():
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = HIP_FP8_E4M3_FNUZ_MAX
else:
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


# Thin wrapper around the extracted CUDA op; this keeps the original API.
_has_native_kernel = False
try:
    # Ensure the extension is loaded (usually handled by __init__.py import, 
    # but we check the ops registry here)
    _quant_op = torch.ops.quant_fp8.sgl_per_token_group_quant_8bit
    _has_native_kernel = True
except (AttributeError, RuntimeError):
    raise RuntimeError("quant_fp8 CUDA extension is not built; build the extension before calling sglang_per_token_group_quant_fp8.")


def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = ceil_align(x_s_mn, 4)
        aligned_k = ceil_align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[: x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if not _has_native_kernel:
        raise RuntimeError(
            "quant_fp8 CUDA extension is not built; "
            "build the extension before calling sglang_per_token_group_quant_fp8."
        )

    if x.shape[0] > 0:
        # We always go through the low-level CUDA kernel path in this extracted version.
        _quant_op(
            x,
            x_q,
            x_s,
            group_size,
            float(eps),
            float(fp8_min),
            float(fp8_max),
            bool(scale_ue8m0),
        )

    return x_q, x_s
