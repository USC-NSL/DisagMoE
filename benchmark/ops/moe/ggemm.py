import torch
import matplotlib.pyplot as plt
import numpy as np
from grouped_gemm.backend import gmm
import os

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
        As = torch.randn(num_experts, n_rows, hidden_size, device=device).contiguous()
        

        # warmp-up
        up_buf = torch.empty((n_rows, intermediate_size * 2), device=device).contiguous()
        down_buf = torch.empty((n_rows, hidden_size), device=device).contiguous()
        for _ in range(2):
            for i in range(num_experts):
                run_expert(As[i], BCs[i], Ds[i], up_buf, down_buf)
        
        # capture cuda graph
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            for i in range(num_experts):
                run_expert(As[i], BCs[i], Ds[i], up_buf, down_buf)
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
        As = torch.randn(num_experts, n_rows, hidden_size, device=device).contiguous()
        
        up_buf = torch.empty((n_rows, intermediate_size * 2), device=device).contiguous()
        down_buf = torch.empty((n_rows, hidden_size), device=device).contiguous()
        
        # warm-up
        for _ in range(2):
            for i in range(num_experts):
                run_expert(As[i], BCs[i], Ds[i], up_buf, down_buf)
        
        # actual measurements
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for i in range(num_experts):
                run_expert(As[i], BCs[i], Ds[i], up_buf, down_buf)
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_sequntial_no_graph.append(np.mean(results_this_batch_size))
    
    # --- grouped GEMM case ---

    # Note that here we always make sure the up/down buffers exactly match
    # the total batch size. In the case the total tokens is less than
    # the buffer sizes, remember to truncate the output tensors of gmm.
    def run_experts_grouped_gemm(As, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes):
        A_flat = As.reshape(-1, As.size(-1)).contiguous()
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

        As = torch.randn(num_experts, n_rows, hidden_size, device=device).contiguous()
        total_rows = num_experts * n_rows
        # torch.empty gives contiguous buffers by default
        up_buf = torch.empty((total_rows, intermediate_size * 2), device=device)
        down_buf = torch.empty((total_rows, hidden_size), device=device)
        batch_sizes = torch.tensor([n_rows] * num_experts, device=device, dtype=torch.int64)

        # warm-up
        for _ in range(2):
            run_experts_grouped_gemm(As, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes)
        
        # capture cuda graph
        stream =  torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            run_experts_grouped_gemm(As, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes)
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
hidden_sizes_k = np.array([6, 4, 5, 7])
hidden_sizes = hidden_sizes_k * 1024
intermediate_sizes_k = np.array([16, 12, 8, 2])
intermediate_sizes = intermediate_sizes_k * 1024
models = ["Mixtral 8x22B", "Mixtral 8x7B", "Llama4", "Deepseek V3 (1/4 #experts)"]
labels = [f"{model}: hidden={h}k, intermediate={i}k" for model, h, i in zip(models, hidden_sizes_k, intermediate_sizes_k)]
num_experts_list = [8, 8, 16, 64]
# num_experts_list = [4, 4, 8, 32]

# Grouped bar chart per model (single figure with subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (hidden_size, intermediate_size, num_experts, label, model) in enumerate(zip(hidden_sizes, intermediate_sizes, num_experts_list, labels, models)):
    row_sizes, seq_means, seq_no_graph_means, grp_means = benchmark_grouped_gemm(hidden_size, intermediate_size, num_experts, label)
    x = np.arange(len(row_sizes))
    width = 0.28
    ax = axes[idx]
    ax.bar(x - width, seq_means, width=width, label="Sequential (graph)")
    ax.bar(x, seq_no_graph_means, width=width, label="Sequential (no graph)")
    ax.bar(x + width, grp_means, width=width, label="Grouped GEMM")
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

plt.tight_layout()
plt.savefig("ggemm_comparison.png", dpi=300)
print("Saved combined plot to ggemm_comparison.png")

if _prof is not None:
    _prof.stop()
    _prof.export_chrome_trace(trace_path)
    print(f"Exported Perfetto-compatible trace to {trace_path}")
