import torch
import time
import numpy as np
from typing import List, Dict, Tuple
import argparse
import os
import sys
import torch.profiler as torch_profiler

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.frontend.datatypes import BatchMetadata, AttentionScheduleBatch
from disagmoe.block_manager.block_manager import CPUBlockManager, GPUBlockManager
from disagmoe_c import BlockManager as BlockManager_C, BatchMetadata as BatchMetadata_C

def prefill_cpu_update_block_table_iter(cpu_mgr: CPUBlockManager, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch) -> float:
    torch.cuda.synchronize()
    t0 = time.time()
    cpu_mgr.update_block_table(meta_c, batch)
    torch.cuda.synchronize()
    return time.time() - t0

def prefill_gpu_update_block_table_iter(gpu_mgr: GPUBlockManager, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch) -> float:
    torch.cuda.synchronize()
    t0 = time.time()
    gpu_mgr.update_block_table(meta_c, batch)
    torch.cuda.synchronize()
    return time.time() - t0

def decode_cpu_update_block_table_iter_layer1(cpu_mgr: CPUBlockManager, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("CPU/update_block_table/decode_layer1_iter"):
        t0 = time.time()
        cpu_mgr.update_block_table(meta_c, batch)
        torch.cuda.synchronize()
    return time.time() - t0

def decode_gpu_update_block_table_iter_layer1(gpu_mgr: GPUBlockManager, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("GPU/update_block_table/decode_layer1_iter"):
        t0 = time.time()
        gpu_mgr.update_block_table(meta_c, batch)
        torch.cuda.synchronize()
    return time.time() - t0

def decode_cpu_pack_flash_attn_iter(cpu_mgr: CPUBlockManager, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch, decode_seq_lens: List[int]) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("CPU/pack_flash_attn_metadata/decode_iter"):
        t0 = time.time()
        _ = cpu_mgr.pack_flash_attn_metadata(meta_c, batch)
        torch.cuda.synchronize()
    return time.time() - t0

def decode_gpu_pack_flash_attn_iter(gpu_mgr: GPUBlockManager, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("GPU/pack_flash_attn_metadata/decode_iter"):
        t0 = time.time()
        _ = gpu_mgr.pack_flash_attn_metadata(meta_c, batch)
        torch.cuda.synchronize()
    return time.time() - t0

def prefill_cpu_setup_iter(cpu_mgr: CPUBlockManager):
    with torch_profiler.record_function("setup/prefill/cpu_reset_iter"):
        cpu_mgr.reset_state()

def prefill_gpu_setup_iter(gpu_mgr: GPUBlockManager):
    with torch_profiler.record_function("setup/prefill/gpu_reset_iter"):
        gpu_mgr.reset_state()

def decode_cpu_setup_layer1_iter(cpu_mgr: CPUBlockManager, batch_size: int, seq_len: int):
    with torch_profiler.record_function("setup/decode_layer1/cpu_reset_and_alloc_iter"):
        cpu_mgr.reset_state()
        for i in range(batch_size):
            cpu_mgr.req_manager.update_decode_seq_lens(i, seq_len)
        for i in range(batch_size):
            cpu_mgr._block_mgr.allocate(i, seq_len)

def decode_gpu_setup_layer1_iter(gpu_mgr: GPUBlockManager, batch_size: int, seq_len: int):
    with torch_profiler.record_function("setup/decode_layer1/gpu_reset_and_alloc_iter"):
        gpu_mgr.reset_state()
        for i in range(batch_size):
            gpu_mgr.decode_seq_lens[i] = seq_len
        new_req_indices = gpu_mgr.req_to_token_pool.alloc(batch_size)
        for i, seq_id in enumerate(range(batch_size)):
            req_idx = new_req_indices[i]
            gpu_mgr.req_to_indice[seq_id] = req_idx
            prefill_kv_locs = gpu_mgr.token_allocator.alloc(seq_len)
            gpu_mgr.req_to_token_pool.write((req_idx, slice(0, seq_len)), prefill_kv_locs)
        batch_req_indices_tensor = torch.tensor(new_req_indices, dtype=torch.int32, device=gpu_mgr.device)
        seq_lens_tensor = torch.full((batch_size,), seq_len, dtype=torch.int32, device=gpu_mgr.device)
        gpu_mgr.req_seq_lens[batch_req_indices_tensor] = seq_lens_tensor

def pack_cpu_setup_iter(cpu_mgr: CPUBlockManager, batch_size: int, seq_len: int) -> List[int]:
    with torch_profiler.record_function("setup/pack_metadata/cpu_reset_and_alloc_iter"):
        cpu_mgr.reset_state()
        for i in range(batch_size):
            cpu_mgr.req_manager.update_decode_seq_lens(i, seq_len)
            cpu_mgr._block_mgr.allocate(i, seq_len)

def pack_gpu_setup_once(gpu_mgr: GPUBlockManager, batch: AttentionScheduleBatch, batch_size: int, seq_len: int):
    with torch_profiler.record_function("setup/pack_metadata/gpu_reset_and_alloc_once"):
        gpu_mgr.reset_state()
        new_req_indices = gpu_mgr.req_to_token_pool.alloc(batch_size)
        for i, seq_id in enumerate(range(batch_size)):
            req_idx = new_req_indices[i]
            gpu_mgr.req_to_indice[seq_id] = req_idx
            prefill_kv_locs = gpu_mgr.token_allocator.alloc(seq_len)
            gpu_mgr.req_to_token_pool.write((req_idx, slice(0, seq_len)), prefill_kv_locs)
            gpu_mgr.decode_seq_lens[seq_id] = seq_len
        batch_req_indices_tensor = torch.tensor(new_req_indices, dtype=torch.int32, device=gpu_mgr.device)
        seq_lens_tensor = torch.full((batch_size,), seq_len, dtype=torch.int32, device=gpu_mgr.device)
        gpu_mgr.req_seq_lens[batch_req_indices_tensor] = seq_lens_tensor
        batch.seq_lens = [seq_len] * batch_size
        batch.seq_lens_tensor = seq_lens_tensor
        batch.req_indices = list(new_req_indices)
        batch.req_indices_tensor = batch_req_indices_tensor

def create_test_metadata(batch_size: int, num_prefill_seqs: int, seq_len: int, layer_id: int = 0) -> Tuple[BatchMetadata_C, AttentionScheduleBatch]:
    """Create test metadata for benchmarking"""
    num_prefill_tokens = num_prefill_seqs
    num_decode_tokens = batch_size - num_prefill_seqs
    
    # Create Python BatchMetadata
    meta_py = BatchMetadata(
        shape=[batch_size, 1024],  # hidden_size
        dtype="bf16",
        layer_id=layer_id,
        req_ids=list(range(batch_size)),
        exp_ids=[0] * batch_size,
        topk_weights=[1.0] * batch_size,
        attn_dp_ranks=[0] * batch_size,
        init_prefill_lens=[seq_len] * num_prefill_seqs + [-1] * num_decode_tokens,
        num_prefill_seqs=num_prefill_seqs,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
    )
    
    # Create dummy data tensor
    data = torch.zeros((batch_size, 1024), dtype=torch.bfloat16, device="cuda:0")
    
    # Create AttentionScheduleBatch from metadata
    batch = AttentionScheduleBatch.build(meta_py, data)
    batch.seq_lens = [seq_len] * batch_size
    
    # Create C++ metadata
    meta_c = BatchMetadata_C()
    meta_c.layer_id = layer_id
    meta_c.shape = [batch_size, 1024]
    meta_c.dtype = "bf16"
    meta_c.req_ids = list(range(batch_size))
    meta_c.exp_ids = [0] * batch_size
    meta_c.topk_weights = [1.0] * batch_size
    meta_c.attn_dp_ranks = [0] * batch_size
    meta_c.init_prefill_lens = [seq_len] * num_prefill_seqs + [-1] * num_decode_tokens
    meta_c.num_prefill_seqs = num_prefill_seqs
    meta_c.num_prefill_tokens = num_prefill_tokens
    meta_c.num_decode_tokens = num_decode_tokens
    
    return meta_c, batch

def benchmark_prefill_only_update_block_table(cpu_mgr: CPUBlockManager, gpu_mgr: GPUBlockManager,
                                            batch_sizes: List[int], seq_len: int, 
                                            num_iterations: int = 100,
                                            profiler=None,
                                            exclude_setup: bool = False) -> Dict[str, List[float]]:
    """Benchmark update_block_table for prefill-only batches of different sizes"""
    results = {
        'batch_sizes': batch_sizes,
        'cpu_times': [],
        'gpu_times': [],
        'speedups': []
    }
    
    print("\n" + "="*60)
    print("BENCHMARK 1: Prefill-only update_block_table")
    print("="*60)
    print(f"{'Batch Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-"*50)
    
    for batch_size in batch_sizes:
        # Create prefill-only metadata (will be recreated each iteration)
        meta_c, _ = create_test_metadata(batch_size, batch_size, seq_len, layer_id=0)
        
        # Warmup (not timed)
        warmup_iters = 2
        for _ in range(warmup_iters):
            prefill_cpu_setup_iter(cpu_mgr)
            _, batch = create_test_metadata(batch_size, batch_size, seq_len, layer_id=0)
            torch.cuda.synchronize()
            cpu_mgr.update_block_table(meta_c, batch)
        for _ in range(warmup_iters):
            prefill_gpu_setup_iter(gpu_mgr)
            _, batch = create_test_metadata(batch_size, batch_size, seq_len, layer_id=0)
            torch.cuda.synchronize()
            gpu_mgr.update_block_table(meta_c, batch)
        
        # CPU version
        cpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset state (not timed)
            prefill_cpu_setup_iter(cpu_mgr)
            # Recreate batch for clean state
            _, batch = create_test_metadata(batch_size, batch_size, seq_len, layer_id=0)
            
            cpu_total_time += prefill_cpu_update_block_table_iter(cpu_mgr, meta_c, batch)
        cpu_time = cpu_total_time / num_iterations * 1000
        
        # GPU version
        gpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset state (not timed)
            prefill_gpu_setup_iter(gpu_mgr)
            # Recreate batch for clean state
            _, batch = create_test_metadata(batch_size, batch_size, seq_len, layer_id=0)
            
            gpu_total_time += prefill_gpu_update_block_table_iter(gpu_mgr, meta_c, batch)
        gpu_time = gpu_total_time / num_iterations * 1000
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results['cpu_times'].append(cpu_time)
        results['gpu_times'].append(gpu_time)
        results['speedups'].append(speedup)
        
        print(f"{batch_size:<12} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<10.2f}")
    
    return results

def benchmark_decode_only_update_block_table(cpu_mgr: CPUBlockManager, gpu_mgr: GPUBlockManager,
                                           batch_sizes: List[int], seq_len: int,
                                           num_iterations: int = 100,
                                           profiler=None,
                                           exclude_setup: bool = False) -> Dict[str, Dict[str, List[float]]]:
    """Benchmark update_block_table for decode-only batches at layer 1 only"""
    results = {
        'batch_sizes': batch_sizes,
        'layer_1': {'cpu_times': [], 'gpu_times': [], 'speedups': []}
    }
    
    print("\n" + "="*60)
    print("BENCHMARK 2: Decode-only update_block_table (Layer 1)")
    print("="*60)
    print(f"{'Batch Size':<12} {'Layer':<6} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-"*60)
    
    for batch_size in batch_sizes:
        # Create decode-only metadata (will be recreated each iteration)
        meta_c, _ = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
        meta_c.layer_id = 1
        
        # Warmup (not timed)
        warmup_iters = 2
        for _ in range(warmup_iters):
            decode_cpu_setup_layer1_iter(cpu_mgr, batch_size, seq_len)
            _, batch = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
            batch.layer_id = 1
            cpu_mgr.update_block_table(meta_c, batch)
            
        for _ in range(warmup_iters):
            decode_gpu_setup_layer1_iter(gpu_mgr, batch_size, seq_len)
            _, batch = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
            batch.layer_id = 1
            gpu_mgr.update_block_table(meta_c, batch)

        torch.cuda.synchronize()

        # CPU version
        cpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset and prepare state (not timed)
            decode_cpu_setup_layer1_iter(cpu_mgr, batch_size, seq_len)
            # Recreate batch for clean state
            _, batch = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
            batch.layer_id = 1
            
            cpu_total_time += decode_cpu_update_block_table_iter_layer1(cpu_mgr, meta_c, batch)
        cpu_time = cpu_total_time / num_iterations * 1000
        
        
        torch.cuda.synchronize()
        
        # GPU version
        gpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset and prepare state (not timed)
            decode_gpu_setup_layer1_iter(gpu_mgr, batch_size, seq_len)
            # Recreate batch for clean state
            _, batch = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
            batch.layer_id = 1
            
            gpu_total_time += decode_gpu_update_block_table_iter_layer1(gpu_mgr, meta_c, batch)
        gpu_time = gpu_total_time / num_iterations * 1000
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results['layer_1']['cpu_times'].append(cpu_time)
        results['layer_1']['gpu_times'].append(gpu_time)
        results['layer_1']['speedups'].append(speedup)
        
        print(f"{batch_size:<12} {1:<6} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<10.2f}")
    
    return results

def benchmark_decode_only_pack_flash_attn_metadata(cpu_mgr: CPUBlockManager, gpu_mgr: GPUBlockManager,
                                                 batch_sizes: List[int], seq_len: int,
                                                 num_iterations: int = 100,
                                                 profiler=None,
                                                 exclude_setup: bool = False) -> Dict[str, List[float]]:
    """Benchmark pack_flash_attn_metadata for decode-only batches with proper init"""
    results = {
        'batch_sizes': batch_sizes,
        'cpu_times': [],
        'gpu_times': [],
        'speedups': []
    }
    
    print("\n" + "="*60)
    print("BENCHMARK 3: Decode-only pack_flash_attn_metadata")
    print("="*60)
    print(f"{'Batch Size':<12} {'CPU (ms)':<12} {'gdrcopy (ms)':<12} {'gdrcopy+rebind (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-"*50)
    
    for batch_size in batch_sizes:
        # Create decode-only metadata at layer 1
        meta_c, batch = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
        _ = pack_cpu_setup_iter(cpu_mgr, batch_size, seq_len)
        
        gpu_total_time = 0.0
        warmup_iters = 2
        
        for _ in range(warmup_iters):
            _ = cpu_mgr.pack_flash_attn_metadata(meta_c, batch)
            
        torch.cuda.synchronize()
        
        def cpu_timed_func():
            
            cpu_total_time = 0.0
        
            for _ in range(num_iterations):
                cpu_total_time += decode_cpu_pack_flash_attn_iter(cpu_mgr, meta_c, batch, [])
                
            cpu_time = cpu_total_time / num_iterations * 1000
            
            return cpu_time
        
        cpu_mgr.use_gdr_copy = False
        cpu_mgr.use_rebind = False
        cpu_time = cpu_timed_func()
        
        cpu_mgr.use_gdr_copy = True
        gdrcopy_time = cpu_timed_func()
        
        cpu_mgr.use_rebind = True
        rebind_time = cpu_timed_func()
        
        _, batch = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
        pack_gpu_setup_once(gpu_mgr, batch, batch_size, seq_len)
        for _ in range(warmup_iters):
            _ = gpu_mgr.pack_flash_attn_metadata(meta_c, batch)
        
        torch.cuda.synchronize()
        
        for _ in range(num_iterations):
            gpu_total_time += decode_gpu_pack_flash_attn_iter(gpu_mgr, meta_c, batch)
        
        torch.cuda.synchronize()
        gpu_time = gpu_total_time / num_iterations * 1000
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results['cpu_times'].append(rebind_time)
        results['gpu_times'].append(gpu_time)
        results['speedups'].append(speedup)
        
        print(f"{batch_size:<12} {cpu_time:<12.3f} {gdrcopy_time:<12.3f} {rebind_time:<12.3f} {gpu_time:<12.3f} {speedup:<10.2f}")
    
    return results

def run_benchmark(batch_sizes: List[int], seq_len: int, num_iterations: int = 100, profiler=None, exclude_setup: bool = False):
    """Run specific benchmarks for update_block_table as requested"""
    
    # Initialize configurations
    model_config = ModelConfig(
        hidden_size=1024,
        num_layers=2,  # Need at least 2 layers for layer 0 vs layer 1 comparison
        num_heads=16,
        num_kv_heads=8,
        num_experts=8,
        intermediate_size=4096,
        dtype=torch.bfloat16,
        ep_size=1,
        tp_size=1,
        dp_size=1,
        layer_ids=[0, 1],
        max_seq_len=4096,
        max_batch_size_attn=512,
    )
    
    cpu_manager_cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.8,
        swap_space=0,
        cache_dtype="auto",
        num_gpu_blocks=10000,
    )
    
    
    # Initialize managers
    cpu_mgr = CPUBlockManager(model_config, cpu_manager_cache_config, 10000, device="cuda:0")
    
    gpu_manager_cache_config = CacheConfig(
        block_size=1,
        gpu_memory_utilization=0.8,
        swap_space=0,
        cache_dtype="auto",
        num_gpu_blocks=160000,
    )
    gpu_mgr = GPUBlockManager(model_config, gpu_manager_cache_config, 10000, device="cuda:0")
    
    print("=" * 80)
    print("KV Cache Management Benchmark: CPU vs GPU")
    print("=" * 80)
    
    all_results = {}
    
    # Benchmark 1: Prefill-only update_block_table
    prefill_results = benchmark_prefill_only_update_block_table(cpu_mgr, gpu_mgr, batch_sizes, seq_len, num_iterations, profiler, exclude_setup)
    all_results['prefill_only'] = prefill_results
    
    # Benchmark 2: Decode-only update_block_table (layer 0 vs layer 1)
    decode_results = benchmark_decode_only_update_block_table(cpu_mgr, gpu_mgr, batch_sizes, seq_len, num_iterations, profiler, exclude_setup)
    all_results['decode_only'] = decode_results
    
    # Benchmark 3: Decode-only pack_flash_attn_metadata
    pack_metadata_results = benchmark_decode_only_pack_flash_attn_metadata(cpu_mgr, gpu_mgr, batch_sizes, seq_len, num_iterations, profiler, exclude_setup)
    all_results['decode_only_pack_metadata'] = pack_metadata_results
    
    torch.cuda.synchronize()
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Prefill-only summary
    prefill_speedups = prefill_results['speedups']
    print(f"Prefill-only Update Block Table:")
    print(f"  Average Speedup: {np.mean(prefill_speedups):.2f}x")
    print(f"  Median Speedup: {np.median(prefill_speedups):.2f}x")
    print(f"  Min Speedup: {np.min(prefill_speedups):.2f}x")
    print(f"  Max Speedup: {np.max(prefill_speedups):.2f}x")
    
    # Decode-only summary
    layer_1_speedups = decode_results['layer_1']['speedups']
    print(f"\nDecode-only Update Block Table (Layer 1):")
    print(f"  Average Speedup: {np.mean(layer_1_speedups):.2f}x")
    print(f"  Median Speedup: {np.median(layer_1_speedups):.2f}x")
    print(f"  Min Speedup: {np.min(layer_1_speedups):.2f}x")
    print(f"  Max Speedup: {np.max(layer_1_speedups):.2f}x")
    
    # Pack metadata summary
    pack_speedups = pack_metadata_results['speedups']
    print(f"\nDecode-only Pack Flash Attn Metadata:")
    print(f"  Average Speedup: {np.mean(pack_speedups):.2f}x")
    print(f"  Median Speedup: {np.median(pack_speedups):.2f}x")
    print(f"  Min Speedup: {np.min(pack_speedups):.2f}x")
    print(f"  Max Speedup: {np.max(pack_speedups):.2f}x")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU KV cache management')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[16, 32, 64, 128, 256],
                       help='Batch sizes to test')
    parser.add_argument('--seq-len', type=int, default=256,
                       help='Sequence length to test')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per test')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results')
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler')
    parser.add_argument('--profile-dir', type=str, default='profiler_traces', help='Directory to store profiler traces')
    parser.add_argument('--profile-warmup', type=int, default=5, help='Profiler warmup steps')
    parser.add_argument('--profile-active', type=int, default=10, help='Profiler active steps')
    parser.add_argument('--profile-repeat', type=int, default=1, help='Profiler repeat cycles')
    parser.add_argument('--profile-exclude-setup', type=bool, default=True, help='Mark setup/warmup ops under setup/* so you can filter them out in the UI')
    
    args = parser.parse_args()
    
    # Set up CUDA
    torch.set_default_device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    
    results = None
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        # One big profiler over the entire benchmark
        with torch_profiler.profile(
            on_trace_ready=torch_profiler.tensorboard_trace_handler(args.profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as p:
            with torch_profiler.record_function("benchmark/run_benchmark"):
                results = run_benchmark(args.batch_sizes, args.seq_len, args.iterations, profiler=p, exclude_setup=args.profile_exclude_setup)
    else:
        results = run_benchmark(args.batch_sizes, args.seq_len, args.iterations)
    
    # Save results if requested
    output_path = args.output
    if output_path is None:
        # Default to profiler directory when profiling; otherwise current dir
        default_dir = args.profile_dir if args.profile else os.getcwd()
        output_path = os.path.join(default_dir, 'benchmark_results.json')
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()