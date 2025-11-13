import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--enable-cudagraph", action="store_true", help="Enable CUDA Graphs for benchmarking")
args = parser.parse_args()
ENABLE_CUDAGRAPH = args.enable_cudagraph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.set_device(device)
torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)

 

@torch.inference_mode()
def benchmark_hidden_size(hidden_size, intermediate_size, label, enable_cudagraph=False):
    
    row_sizes = np.arange(1, 512, 1) # 1 to 512
    # row_sizes = np.concatenate((np.arange(128, 512, 32), np.arange(512, 2048 + 1, 128)))
    
    num_runs = 20
    num_repeats = 5

    BC = torch.randn(hidden_size, intermediate_size * 2, device=device)
    D = torch.randn(intermediate_size, hidden_size, device=device)
        
    def run_fuse_w13(x):
        t1 = torch.matmul(x, BC)
        t2 = t1[:, :intermediate_size] * t1[:, intermediate_size:]
        _ = torch.matmul(t2, D)
        
    results = []
    
    for rows in row_sizes:
        batch_size_results = []
        A = torch.randn(rows, hidden_size, device=device)
        for _ in range(2):
            run_fuse_w13(A)
        
        if enable_cudagraph:
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=stream):
                run_fuse_w13(A)
            torch.cuda.synchronize(device)
            for _ in range(2):
                graph.replay()
            
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(num_runs):
                if enable_cudagraph:
                    graph.replay()
                else:
                    run_fuse_w13(A)
            end.record()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_time = total_time / num_runs
            batch_size_results.append(avg_time)
        results.append(np.mean(batch_size_results))

    torch.cuda.empty_cache()
    
    # Return batch sizes and average times (ms)
    return row_sizes, np.array(results)
    
hidden_sizes_k = np.array([6, 4, 5, 7])
hidden_sizes = hidden_sizes_k * 1024
intermediate_sizes_k = np.array([16, 12, 8, 2])
intermediate_sizes = intermediate_sizes_k * 1024
models = ["Mixtral 8x22B", "Mixtral 8x7B", "Llama4", "Deepseek V3"]
labels = [f"{model}: hidden={h}k, intermediate={i}k" for model, h, i in zip(models, hidden_sizes_k, intermediate_sizes_k)]

 
datasets = []
for hidden_size, intermediate_size, label in zip(hidden_sizes, intermediate_sizes, labels):
    row_sizes, times_ms = benchmark_hidden_size(hidden_size, intermediate_size, label, enable_cudagraph=ENABLE_CUDAGRAPH)
    datasets.append((row_sizes, times_ms, label))
    # Write per-model CSV
    output_dir = "expert_costs_profiles"
    os.makedirs(output_dir, exist_ok=True)
    safe_label = re.sub(r'[^A-Za-z0-9._-]+', '_', label)
    csv_path = os.path.join(output_dir, f"{safe_label}.csv")
    np.savetxt(csv_path, np.column_stack((row_sizes, times_ms)), delimiter=",", header="batch_size,avg_time_ms", comments="", fmt=['%d', '%.6f'])

# Time cost vs. batch size
plt.figure(figsize=(10, 6))
for row_sizes, times_ms, label in datasets:
    plt.plot(row_sizes, times_ms, marker="o", label=label)
title_suffix = " (CUDA Graphs: ON)" if ENABLE_CUDAGRAPH else " (CUDA Graphs: OFF)"
plt.title("Time Cost vs. Batch Size" + title_suffix)
plt.xlabel("batch size")
plt.ylabel("Avg Execution Time (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("expert_cost_fuse_w13.png", dpi=300)

# Throughput vs. batch size (samples per second)
plt.figure(figsize=(10, 6))
for row_sizes, times_ms, label in datasets:
    throughput = (row_sizes * 1000.0) / np.maximum(times_ms, 1e-9)  # samples/s
    plt.plot(row_sizes, throughput, marker="o", label=label)
plt.title("Throughput vs. Batch Size" + title_suffix)
plt.xlabel("batch size")
plt.ylabel("Throughput (samples/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("expert_throughput_fuse_w13.png", dpi=300)