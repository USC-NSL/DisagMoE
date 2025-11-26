
import torch
from disagmoe.ops.fp8_quantizer.fp8_quant import sglang_per_token_group_quant_fp8

def example_usage() -> None:
    """Minimal example demonstrating usage of sglang_per_token_group_quant_fp8."""
    if not torch.cuda.is_available():
        print("CUDA not available; skipping example.")
        return

    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    group_size = 32

    x_q, x_s = sglang_per_token_group_quant_fp8(
        x,
        group_size=group_size,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )

    print("Input shape:", x.shape)
    print("Quantized shape:", x_q.shape, "dtype:", x_q.dtype)
    print("Scale shape:", x_s.shape, "dtype:", x_s.dtype)

if __name__ == "__main__":
    example_usage()

