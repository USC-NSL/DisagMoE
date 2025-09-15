#!/usr/bin/env python3
"""
Example usage of the KV cache management benchmark
"""

import torch
import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from prepare_batch import run_benchmark

def run_memory_intensive_benchmark():
    """Run benchmark with memory-intensive configurations"""
    print("Running memory-intensive benchmark...")
    
    batch_sizes = [64, 128, 256, 512]
    seq_lens = [512, 1024, 2048]
    num_iterations = 50
    
    results = run_benchmark(batch_sizes, seq_lens, num_iterations)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"memory_intensive_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

def run_throughput_benchmark():
    """Run benchmark focused on throughput (many small batches)"""
    print("Running throughput benchmark...")
    
    batch_sizes = [8, 16, 32, 64]
    seq_lens = [64, 128, 256]
    num_iterations = 200  # More iterations for better statistics
    
    results = run_benchmark(batch_sizes, seq_lens, num_iterations)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"throughput_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

def run_latency_benchmark():
    """Run benchmark focused on latency (single batch processing)"""
    print("Running latency benchmark...")
    
    batch_sizes = [1, 2, 4, 8, 16]
    seq_lens = [32, 64, 128, 256, 512]
    num_iterations = 1000  # Many iterations for accurate timing
    
    results = run_benchmark(batch_sizes, seq_lens, num_iterations)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"latency_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

def analyze_results(results):
    """Analyze benchmark results and print insights"""
    print("\n" + "="*60)
    print("BENCHMARK ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    update_speedups = [r['update_speedup'] for r in results]
    pack_speedups = [r['pack_speedup'] for r in results]
    
    print(f"Update Block Table Performance:")
    print(f"  Average speedup: {sum(update_speedups)/len(update_speedups):.2f}x")
    print(f"  Best speedup: {max(update_speedups):.2f}x")
    print(f"  Worst speedup: {min(update_speedups):.2f}x")
    
    print(f"\nPack Flash Attn Metadata Performance:")
    print(f"  Average speedup: {sum(pack_speedups)/len(pack_speedups):.2f}x")
    print(f"  Best speedup: {max(pack_speedups):.2f}x")
    print(f"  Worst speedup: {min(pack_speedups):.2f}x")
    
    # Find best and worst cases
    best_update = max(results, key=lambda x: x['update_speedup'])
    worst_update = min(results, key=lambda x: x['update_speedup'])
    
    print(f"\nBest Update Performance:")
    print(f"  Batch size: {best_update['batch_size']}, Seq len: {best_update['seq_len']}")
    print(f"  Speedup: {best_update['update_speedup']:.2f}x")
    
    print(f"\nWorst Update Performance:")
    print(f"  Batch size: {worst_update['batch_size']}, Seq len: {worst_update['seq_len']}")
    print(f"  Speedup: {worst_update['update_speedup']:.2f}x")

def main():
    """Main function to run different benchmark scenarios"""
    print("KV Cache Management Benchmark - Example Usage")
    print("="*50)
    
    # Set up CUDA
    torch.set_default_device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("Warning: Low GPU memory detected. Consider reducing batch sizes.")
    else:
        print("Error: CUDA not available!")
        return
    
    # Run different benchmark scenarios
    scenarios = [
        ("Memory Intensive", run_memory_intensive_benchmark),
        ("Throughput", run_throughput_benchmark),
        ("Latency", run_latency_benchmark),
    ]
    
    all_results = {}
    
    for scenario_name, scenario_func in scenarios:
        print(f"\n{'='*20} {scenario_name} {'='*20}")
        try:
            results = scenario_func()
            all_results[scenario_name] = results
            analyze_results(results)
        except Exception as e:
            print(f"Error in {scenario_name} benchmark: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_output = f"combined_benchmark_results_{timestamp}.json"
    
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {combined_output}")
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
