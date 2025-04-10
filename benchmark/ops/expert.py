import torch
import matplotlib.pyplot as plt
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

plt.figure(figsize=(10, 6))

@torch.inference_mode()
def benchmark_hidden_size(hidden_size, intermediate_size):
    row_sizes = np.concatenate((np.arange(4, 128 + 1, 4), np.arange(128, 512 + 1, 32)))
    num_runs = 20
    num_repeats = 5

    B = torch.randn(hidden_size, intermediate_size, device=device)
    C = torch.randn(hidden_size, intermediate_size, device=device)
    D = torch.randn(intermediate_size, hidden_size, device=device)

    
    BC = torch.randn(hidden_size, intermediate_size * 2, device=device)
    D = torch.randn(intermediate_size, hidden_size, device=device)
    
    def run(x):
        t1 = torch.matmul(x, B)
        t2 = torch.matmul(x, C)
        t = t1 * t2
        _ = torch.matmul(t, D)
        
    def run_fuse_w13(x):
        t1 = torch.matmul(x, BC)
        t2 = t1[:, :intermediate_size] * t1[:, intermediate_size:]
        _ = torch.matmul(t2, D)
        
    # Warm-up
    for _ in range(10):
        _ = torch.matmul(torch.randn(512, hidden_size, device=device), B)

    results = {rows: [] for rows in row_sizes}

    for _ in range(num_repeats):
        random.shuffle(row_sizes)
        for rows in row_sizes:
            A = torch.randn(rows, hidden_size, device=device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_runs):
                run_fuse_w13(A)
            end.record()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_time = total_time / num_runs
            results[rows].append(avg_time)

    # Average over repeats
    avg_times = {rows: np.mean(times) for rows, times in results.items()}
    
    # Plot
    plt.plot(list(avg_times.keys()), list(avg_times.values()), marker="o", label=f"hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    
hidden_size_k = np.array([4, 5, 6])
hidden_size_k = hidden_size_k * 1024
intermediate_size_k = np.array([12, 8, 16])
intermediate_size_k = intermediate_size_k * 1024

for hidden_size, intermediate_size in zip(hidden_size_k, intermediate_size_k):
    benchmark_hidden_size(hidden_size, intermediate_size)


plt.title("Time Cost vs. Batch Size")
plt.xlabel("batch size")
plt.ylabel("Avg Execution Time (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("expert_cost_fuse_w13.png", dpi=300)