import torch
import matplotlib.pyplot as plt
import numpy as np
from sgl_kernel import fp8_scaled_mm
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.set_device(device)
torch.set_default_device(device)
# Enforce FP8 benchmarking only (use BF16 for generation/accumulation as needed)
DTYPE_FP8 = torch.float8_e4m3fn
ACCUM_DTYPE = torch.bfloat16
print("Benchmark dtype: FP8 (compute outputs in BF16)")

# Minimal Perfetto trace enablement via env var PERFETTO_TRACE
trace_path = os.environ.get("PERFETTO_TRACE")
_prof = None
if trace_path:
    _prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    )
    _prof.start()

@torch.inference_mode()
def alloc_expert_weights(
    hidden_size,
    intermediate_size,
    num_experts,
    device,
    dtype=DTYPE_FP8,
):
    # Generate weights in BF16 then store as FP8
    BCs_bf16 = torch.randn(
        num_experts, hidden_size, intermediate_size * 2, device=device, dtype=ACCUM_DTYPE
    ).contiguous()
    Ds_bf16 = torch.randn(
        num_experts, intermediate_size, hidden_size, device=device, dtype=ACCUM_DTYPE
    ).contiguous()
    BCs_fp8 = BCs_bf16.to(DTYPE_FP8).contiguous()
    Ds_fp8 = Ds_bf16.to(DTYPE_FP8).contiguous()
    # Ensure per-expert weight matrices are column-major in memory for SGL kernel
    # This preserves logical shapes: BCs: (H, 2I), Ds: (I, H)
    BCs_fp8 = BCs_fp8.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    Ds_fp8 = Ds_fp8.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    return BCs_fp8, Ds_fp8

 

@torch.inference_mode()
def benchmark_bf16_end_to_end(hidden_size, intermediate_size, num_experts, label):
    # Same batch sizes
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512])

    num_repeats = 5

    # Allocate BF16 weights directly (end-to-end BF16)
    BCs = torch.randn(
        num_experts, hidden_size, intermediate_size * 2, device=device, dtype=ACCUM_DTYPE
    ).contiguous()
    Ds = torch.randn(
        num_experts, intermediate_size, hidden_size, device=device, dtype=ACCUM_DTYPE
    ).contiguous()

    def run_expert(x, BC, D, up_buf, down_buf):
        torch.matmul(x, BC, out=up_buf)
        up1 = up_buf[:, :intermediate_size]
        up3 = up_buf[:, intermediate_size:]
        up1 *= up3
        _ = torch.matmul(up1, D, out=down_buf)

    results_bf16_e2e = []

    for n_rows in row_sizes:
        results_this_batch_size = []
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device, dtype=ACCUM_DTYPE).contiguous() for bs in batch_sizes_i]
        up_bufs = [torch.empty((bs, intermediate_size * 2), device=device, dtype=ACCUM_DTYPE).contiguous() for bs in batch_sizes_i]
        down_bufs = [torch.empty((bs, hidden_size), device=device, dtype=ACCUM_DTYPE).contiguous() for bs in batch_sizes_i]

        # warm-up
        for _ in range(2):
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])

        # capture cuda graph
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])
        torch.cuda.synchronize(device)

        # warm-up graph
        for _ in range(2):
            graph.replay()

        # timed replays
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_bf16_e2e.append(np.mean(results_this_batch_size))

    torch.cuda.empty_cache()
    return row_sizes, results_bf16_e2e

@torch.inference_mode()
def benchmark_sgl_fp8_kernel(hidden_size, intermediate_size, num_experts, label):

    # Use the same batch sizes as other benchmarks
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512])

    num_repeats = 5

    # Expert weights in FP8
    BCs_fp8, Ds_fp8 = alloc_expert_weights(hidden_size, intermediate_size, num_experts, device, dtype=DTYPE_FP8)
    # Scales: use per-column scales for mat_b as required by SGL kernel
    # For BCs: (H, 2I) -> scales_b size = 2I; For Ds: (I, H) -> scales_b size = H
    scales_b_up = [torch.ones(BCs_fp8[i].shape[1], device=device, dtype=torch.float32) for i in range(num_experts)]
    scales_b_down = [torch.ones(Ds_fp8[i].shape[1], device=device, dtype=torch.float32) for i in range(num_experts)]
    # Keep mat_a scale as scalar 1.0 (allowed by kernel)
    one_scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    results_sgl = []

    for n_rows in row_sizes:
        results_this_batch_size = []

        # Inputs: per-expert tokens (generate in BF16, then convert to FP8)
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list_bf16 = [torch.randn(bs, hidden_size, device=device, dtype=ACCUM_DTYPE).contiguous() for bs in batch_sizes_i]
        As_list_fp8 = [a.to(DTYPE_FP8).contiguous() for a in As_list_bf16]
        # Per-row activation scales for mat_a (length = batch size per expert)
        scales_a = [torch.ones(As_list_fp8[i].shape[0], device=device, dtype=torch.float32) for i in range(num_experts)]

        # Warm-up to trigger JIT compilation and stabilize runtime
        def run_once():
            for i in range(num_experts):
                # GEMM-1: (bs, H) x (H, 2I) -> (bs, 2I) in BF16
                up = fp8_scaled_mm(
                    As_list_fp8[i],
                    BCs_fp8[i],
                    scales_a[i],
                    scales_b_up[i],
                    ACCUM_DTYPE,
                    None,
                )
                up_1 = up[:, :intermediate_size]
                up_3 = up[:, intermediate_size:]
                glu = (up_1 * up_3).contiguous()
                # Quantize activation back to FP8 for GEMM-2
                glu_fp8 = glu.to(DTYPE_FP8).contiguous()
                _ = fp8_scaled_mm(
                    glu_fp8,
                    Ds_fp8[i],
                    scales_a[i],
                    scales_b_down[i],
                    ACCUM_DTYPE,
                    None,
                )

        # Warm-up a few times
        for _ in range(2):
            run_once()

        # Capture CUDA graph for two FP8 GEMMs + activation
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            run_once()
        torch.cuda.synchronize(device)

        # Warm-up graph
        for _ in range(2):
            graph.replay()

        # Timed replays
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_sgl.append(np.mean(results_this_batch_size))

    torch.cuda.empty_cache()

    return row_sizes, results_sgl


hidden_sizes_k = np.array([2, 4])
hidden_sizes = hidden_sizes_k * 1024
intermediate_sizes = [768, 1536]
models = ["Qwen3-30B", "Qwen3-235B"]
labels = [f"{model}: hidden={h}k, intermediate={i}" for model, h, i in zip(models, hidden_sizes_k, intermediate_sizes)]
num_experts_list = [8, 8]
# expert_batch_size_ratios = [1, 0.5, 0.1, 0.2, 0.8, 1.5, 1.9, 0.7] # this must equals to num_experts
expert_batch_size_ratios = [1] * 8

# Grouped bar chart per model (single figure with subplots)
n_models = len(models)
n_cols = 2
n_rows = int(np.ceil(n_models / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
axes = np.array(axes).ravel()

flops_series_per_model = []
time_series_bf16_e2e = []
time_series_sgl = []
model_names = []
model_colors = []

for idx, (hidden_size, intermediate_size, num_experts, label, model) in enumerate(zip(hidden_sizes, intermediate_sizes, num_experts_list, labels, models)):
    row_sizes_bf16, bf16_e2e_means = benchmark_bf16_end_to_end(hidden_size, intermediate_size, num_experts, label)
    row_sizes_sgl, sgl_means = benchmark_sgl_fp8_kernel(hidden_size, intermediate_size, num_experts, label)
    assert np.array_equal(row_sizes_bf16, row_sizes_sgl), "Row sizes mismatch between benchmarks"
    x = np.arange(len(row_sizes_bf16))
    width = 0.35
    ax = axes[idx]
    ax.bar(x - 0.5*width, bf16_e2e_means, width=width, label="Torch matmul (BF16 graph)")
    ax.bar(x + 0.5*width, sgl_means, width=width, label="SGL FP8 kernel")
    # Use sparse xticks for readability
    if len(row_sizes_bf16) > 12:
        tick_idx = np.linspace(0, len(row_sizes_bf16) - 1, num=12, dtype=int)
    else:
        tick_idx = np.arange(len(row_sizes_bf16))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([int(row_sizes_bf16[i]) for i in tick_idx])
    ax.set_title(f"{label}, experts={num_experts}")
    ax.set_xlabel("per-expert batch size")
    ax.set_ylabel("Avg Execution Time (ms)")
    ax.legend()

    # Collect FLOPs vs time data (for all methods)
    flops_total = 6.0 * float(num_experts) * float(hidden_size) * float(intermediate_size) * row_sizes_bf16.astype(np.float64)
    flops_series_per_model.append(flops_total / 1e9)  # in GFLOPs
    time_series_bf16_e2e.append(np.array(bf16_e2e_means, dtype=np.float64))
    time_series_sgl.append(np.array(sgl_means, dtype=np.float64))
    model_names.append(model)
    model_colors.append(plt.get_cmap('tab10')(idx))

# Hide any unused subplots
for ax in axes[len(models):]:
    ax.set_visible(False)

plt.tight_layout()
_cmp_path = f"bf16_vs_sgl_fp8.png"
plt.savefig(_cmp_path, dpi=300)
print(f"Saved combined plot to {_cmp_path}")

# Create FLOPs vs Time line plot
fig2, ax2 = plt.subplots(figsize=(9, 6))
# Plot per model with consistent color; two methods: Torch BF16 vs SGL FP8
for flops_g, t_bf16, t_sgl, name, color in zip(
    flops_series_per_model,
    time_series_bf16_e2e,
    time_series_sgl,
    model_names,
    model_colors,
):
    ax2.plot(flops_g, t_bf16, marker='o', linestyle='-', color=color, label=f"{name} - Torch BF16 matmul")
    ax2.plot(flops_g, t_sgl, marker='^', linestyle='--', color=color, alpha=0.9, label=f"{name} - SGL FP8 kernel")
ax2.set_xlabel("Estimated FLOPs per batch (GFLOPs)")
ax2.set_ylabel("Avg Execution Time (ms)")
ax2.set_title("Time vs Estimated FLOPs (Torch BF16 vs SGL FP8)")
ax2.legend(ncol=2, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.3)
fig2.tight_layout()
_flops_path = f"ggemm_flops_vs_time_bf16.png"
fig2.savefig(_flops_path, dpi=300)
print(f"Saved FLOPs vs time plot to {_flops_path}")

if _prof is not None:
    _prof.stop()
    _prof.export_chrome_trace(trace_path)
    print(f"Exported Perfetto-compatible trace to {trace_path}")
