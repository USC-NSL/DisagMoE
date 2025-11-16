import torch
import matplotlib.pyplot as plt
import numpy as np
from grouped_gemm.backend import gmm
import os
from triton_moe_demo import (
    _moe_align_block_size,
    _invoke_fused_moe_kernel,
    _silu_and_mul,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.set_device(device)
torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)

# Minimal Perfetto trace enablement via env var PERFETTO_TRACE
trace_path = os.environ.get("PERFETTO_TRACE")
_prof = None
if trace_path:
    _prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    )
    _prof.start()

@torch.inference_mode()
def alloc_expert_weights(hidden_size, intermediate_size, num_experts,
                         device, dtype=torch.bfloat16):
    BCs = torch.randn(num_experts, hidden_size, intermediate_size * 2, device=device, dtype=dtype).contiguous()
    Ds = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype).contiguous()
    return BCs, Ds

 

@torch.inference_mode()
def benchmark_grouped_gemm(hidden_size, intermediate_size, num_experts, label):

    # this is just batch sizes we test
    # row_sizes = np.concatenate((np.arange(4, 128, 4), np.arange(128, 512 + 1, 32)))
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512])

    num_repeats = 5

    # BC = torch.randn(hidden_size, intermediate_size * 2, device=device)
    # D = torch.randn(intermediate_size, hidden_size, device=device)

    BCs, Ds = alloc_expert_weights(hidden_size, intermediate_size, num_experts, device)

    # --- Sequntial CUDA graph case ---

    # GLU here, not SwiGLU; computational load are roughly the same
    def run_expert(x, BC, D, up_buf, down_buf):
        torch.matmul(x, BC, out=up_buf)
        up1 = up_buf[:, :intermediate_size]
        up3 = up_buf[:, intermediate_size:]
        up1 *= up3
        _ = torch.matmul(up1, D, out=down_buf)

    results_sequntial = []

    for n_rows in row_sizes:
        results_this_batch_size = []
        # per-expert variable batch sizes
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device).contiguous() for bs in batch_sizes_i]
        up_bufs = [torch.empty((bs, intermediate_size * 2), device=device).contiguous() for bs in batch_sizes_i]
        down_bufs = [torch.empty((bs, hidden_size), device=device).contiguous() for bs in batch_sizes_i]

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
        
        # actual measurments
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_sequntial.append(np.mean(results_this_batch_size))

    # --- sequntial non-CUDAGraph case

    results_sequntial_no_graph = []

    for n_rows in row_sizes:
        results_this_batch_size = []
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device).contiguous() for bs in batch_sizes_i]
        up_bufs = [torch.empty((bs, intermediate_size * 2), device=device).contiguous() for bs in batch_sizes_i]
        down_bufs = [torch.empty((bs, hidden_size), device=device).contiguous() for bs in batch_sizes_i]
        
        # warm-up
        for _ in range(2):
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])
        
        # actual measurements
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_sequntial_no_graph.append(np.mean(results_this_batch_size))
    
    # --- grouped GEMM case ---

    # Note that here we always make sure the up/down buffers exactly match
    # the total batch size. In the case the total tokens is less than
    # the buffer sizes, remember to truncate the output tensors of gmm.
    def run_experts_grouped_gemm(As_list, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes):
        A_flat = torch.cat(As_list, dim=0).contiguous()
        up = gmm(A_flat, BCs, batch_sizes, c=up_buf)
        # NOTE: the up_1 slice is NOT contiguous
        # Compute GLU into a new contiguous buffer to satisfy gmm's contiguity requirement
        up_1 = up[:, :intermediate_size]
        up_3 = up[:, intermediate_size:]
        glu = (up_1 * up_3).contiguous()
        _ = gmm(glu, Ds, batch_sizes, c=down_buf)
    
    results_grouped_gemm = []

    for n_rows in row_sizes:
        results_this_batch_size = []

        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device).contiguous() for bs in batch_sizes_i]
        total_rows = int(sum(batch_sizes_i))
        # torch.empty gives contiguous buffers by default
        up_buf = torch.empty((total_rows, intermediate_size * 2), device=device)
        down_buf = torch.empty((total_rows, hidden_size), device=device)
        batch_sizes = torch.tensor(batch_sizes_i, device=device, dtype=torch.int64)

        # warm-up
        for _ in range(2):
            run_experts_grouped_gemm(As_list, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes)
        
        # capture cuda graph
        stream =  torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            run_experts_grouped_gemm(As_list, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes)
        torch.cuda.synchronize(device)

        # warm-up graph
        for _ in range(2):
            graph.replay()
        
        # actual measurements
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_grouped_gemm.append(np.mean(results_this_batch_size))
        
    torch.cuda.empty_cache()

    # No plotting here; just return results
    return row_sizes, results_sequntial, results_sequntial_no_graph, results_grouped_gemm


@torch.inference_mode()
def benchmark_triton_fused_moe(hidden_size, intermediate_size, num_experts, label):

    # Use the same batch sizes as other benchmarks
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512])

    num_repeats = 5

    # Expert weights in the layout expected by the Triton demo helpers
    # BCs: (E, H, 2I)  -> transpose to (E, 2I, H) for GEMM-1
    # Ds:  (E, I, H)   -> transpose to (E, H, I)  for GEMM-2 (pre-transpose)
    BCs, Ds = alloc_expert_weights(hidden_size, intermediate_size, num_experts, device)
    B1 = BCs.transpose(1, 2).contiguous()                 # (E, 2I, H)
    w2 = Ds.transpose(1, 2).contiguous()                  # (E, H, I)

    results_triton_fused = []

    for n_rows in row_sizes:
        results_this_batch_size = []

        # Inputs: per-expert tokens with variable sizes, then flattened to (M, H)
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device).contiguous() for bs in batch_sizes_i]
        A_flat = torch.cat(As_list, dim=0).contiguous()   # (M, H)

        # Top-1 routing mapping each token to its expert deterministically
        top_k = 1
        m = int(sum(batch_sizes_i))
        topk_ids = torch.empty((m, top_k), device=device, dtype=torch.int32)
        # Fill expert ids per contiguous segment
        start = 0
        for e, bs in enumerate(batch_sizes_i):
            end = start + int(bs)
            topk_ids[start:end, 0] = int(e)
            start = end
        topk_weights = torch.ones((m, top_k), device=device, dtype=torch.bfloat16)

        # Align tokens into kernel-friendly blocks
        BLOCK_SIZE_M = 64
        sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
            topk_ids, BLOCK_SIZE_M, num_experts
        )
        EM = int(num_tokens_post_padded.item())
        if EM == 0:
            results_triton_fused.append(0.0)
            continue

        # Output buffers; for GEMM-2, use B as (E, N=k, K=d_ff) = w2
        c1 = torch.empty((EM, 2 * intermediate_size), device=device, dtype=torch.bfloat16)
        inter_buf = torch.empty((EM, intermediate_size), device=device, dtype=torch.bfloat16)
        c2 = torch.empty((EM, hidden_size), device=device, dtype=torch.bfloat16)

        # Warm-up to trigger JIT compilation and stabilize runtime
        def run_once():
            _invoke_fused_moe_kernel(
                A_flat,
                B1,
                c1,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k=top_k,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8,
                mul_routed_weight=False,
            )
            inter_buf.copy_(_silu_and_mul(c1))
            _invoke_fused_moe_kernel(
                inter_buf,
                w2,
                c2,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k=top_k,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8,
                mul_routed_weight=False,
            )

        # Warm-up a few times
        for _ in range(2):
            run_once()

        # Capture CUDA graph for the two Triton GEMMs + activation
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
        results_triton_fused.append(np.mean(results_this_batch_size))

    torch.cuda.empty_cache()

    return row_sizes, results_triton_fused


hidden_sizes_k = np.array([6, 4, 7, 2, 4])
hidden_sizes = hidden_sizes_k * 1024
intermediate_sizes_k = np.array([16, 12, 2, 6, 12])
intermediate_sizes = intermediate_sizes_k * 1024
models = ["Mixtral 8x22B", "Mixtral 8x7B", "Deepseek V3", "Qwen3-30B", "Qwen3-235B"]
labels = [f"{model}: hidden={h}k, intermediate={i}k" for model, h, i in zip(models, hidden_sizes_k, intermediate_sizes_k)]
num_experts_list = [8, 8, 8, 8, 8]
expert_batch_size_ratios = [1, 0.5, 0.1, 0.2, 0.8, 1.5, 1.9, 0.7] # this must equals to num_experts

# Grouped bar chart per model (single figure with subplots)
n_models = len(models)
n_cols = 2
n_rows = int(np.ceil(n_models / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
axes = np.array(axes).ravel()

flops_series_per_model = []
time_series_grouped = []
time_series_seq_no_graph = []
time_series_triton = []
model_names = []
model_colors = []

for idx, (hidden_size, intermediate_size, num_experts, label, model) in enumerate(zip(hidden_sizes, intermediate_sizes, num_experts_list, labels, models)):
    row_sizes, seq_means, seq_no_graph_means, grp_means = benchmark_grouped_gemm(hidden_size, intermediate_size, num_experts, label)
    row_sizes_t, triton_means = benchmark_triton_fused_moe(hidden_size, intermediate_size, num_experts, label)
    assert np.array_equal(row_sizes, row_sizes_t), "Row sizes mismatch between benchmarks"
    x = np.arange(len(row_sizes))
    width = 0.22
    ax = axes[idx]
    ax.bar(x - 1.5*width, seq_means, width=width, label="Sequential (graph)")
    ax.bar(x - 0.5*width, seq_no_graph_means, width=width, label="Sequential (no graph)")
    ax.bar(x + 0.5*width, grp_means, width=width, label="Grouped GEMM")
    ax.bar(x + 1.5*width, triton_means, width=width, label="Triton Fused MoE")
    # Use sparse xticks for readability
    if len(row_sizes) > 12:
        tick_idx = np.linspace(0, len(row_sizes) - 1, num=12, dtype=int)
    else:
        tick_idx = np.arange(len(row_sizes))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([int(row_sizes[i]) for i in tick_idx])
    ax.set_title(f"{label}, experts={num_experts}")
    ax.set_xlabel("per-expert batch size")
    ax.set_ylabel("Avg Execution Time (ms)")
    ax.legend()

    # Collect FLOPs vs time data (for all methods)
    flops_total = 6.0 * float(num_experts) * float(hidden_size) * float(intermediate_size) * row_sizes.astype(np.float64)
    flops_series_per_model.append(flops_total / 1e9)  # in GFLOPs
    time_series_grouped.append(np.array(grp_means, dtype=np.float64))
    time_series_seq_no_graph.append(np.array(seq_no_graph_means, dtype=np.float64))
    time_series_triton.append(np.array(triton_means, dtype=np.float64))
    model_names.append(model)
    model_colors.append(plt.get_cmap('tab10')(idx))

# Hide any unused subplots
for ax in axes[len(models):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("ggemm_comparison.png", dpi=300)
print("Saved combined plot to ggemm_comparison.png")

# Create FLOPs vs Time line plot (all methods on one figure)
fig2, ax2 = plt.subplots(figsize=(9, 6))
# Plot per model with consistent color; two methods: Grouped GEMM vs Sequential (no graph)
for flops_g, t_grp, t_sng, t_tri, name, color in zip(
    flops_series_per_model,
    time_series_grouped,
    time_series_seq_no_graph,
    time_series_triton,
    model_names,
    model_colors,
):
    ax2.plot(flops_g, t_grp, marker='o', linestyle='-', color=color, label=f"{name} - Grouped GEMM")
    ax2.plot(flops_g, t_sng, marker='s', linestyle=':', color=color, alpha=0.9, label=f"{name} - Sequential (no graph)")
    ax2.plot(flops_g, t_tri, marker='^', linestyle='--', color=color, alpha=0.9, label=f"{name} - Triton Fused MoE")
ax2.set_xlabel("Estimated FLOPs per batch (GFLOPs)")
ax2.set_ylabel("Avg Execution Time (ms)")
ax2.set_title("Time vs Estimated FLOPs (Grouped GEMM vs Separate Kernels)")
ax2.legend(ncol=2, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.3)
fig2.tight_layout()
fig2.savefig("ggemm_flops_vs_time.png", dpi=300)
print("Saved FLOPs vs time plot to ggemm_flops_vs_time.png")

if _prof is not None:
    _prof.stop()
    _prof.export_chrome_trace(trace_path)
    print(f"Exported Perfetto-compatible trace to {trace_path}")
