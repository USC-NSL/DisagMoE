import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.set_device(device)
torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})

@torch.inference_mode()
def benchmark_hidden_size(hidden_size, intermediate_size, label):
    
    row_sizes = np.concatenate((np.arange(4, 128, 4), np.arange(128, 512 + 1, 32)))
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
                graph.replay()
                # run_fuse_w13(A)
            end.record()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_time = total_time / num_runs
            tput = rows / avg_time * 1000
            batch_size_results.append(tput)
        results.append(np.mean(batch_size_results))

    torch.cuda.empty_cache()
    
    # Plot
    plt.plot(row_sizes, results, marker="o", label=label)
    
hidden_sizes_k = np.array([6, 4, 5])
hidden_sizes = hidden_sizes_k * 1024
intermediate_sizes_k = np.array([16, 12, 8])
intermediate_sizes = intermediate_sizes_k * 1024
models = ["Mixtral 8x22B", "Mixtral 8x7B", "Llama4"]
labels = [f"{model}: hidden={h}k, intermediate={i}k" for model, h, i in zip(models, hidden_sizes_k, intermediate_sizes_k)]

for hidden_size, intermediate_size, label in zip(hidden_sizes, intermediate_sizes, labels):
    benchmark_hidden_size(hidden_size, intermediate_size, label)

plt.title("Expert Throughput vs. Batch Size")
plt.xlabel("batch size")
plt.ylabel("throughput tokens/s")

yticks_idx = np.arange(0, 1000000, 200000)
yticks_k = [ f"{int(i / 1000)}k" for i in yticks_idx]
plt.yticks(yticks_idx, yticks_k)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("expert_cost_fuse_w13.png", dpi=300)