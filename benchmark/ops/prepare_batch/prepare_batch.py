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
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from disagmoe.executor.executor import AttnExecutor
from disagmoe.executor.block_manager import TokenToKVPoolAllocator, ReqToTokenPool
from disagmoe.models.utils import make_dummy_meta
from disagmoe_c import BlockManager as BlockManager_C, AttentionBatchMetadata as AttentionBatchMetadata_C
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

class CPUBlockManager:
    """CPU-based block manager for comparison - follows original implementation"""
    def __init__(self, block_size: int, num_gpu_blocks: int, num_reserved_blocks: int, model_config, device: str = "cuda"):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_reserved_blocks = num_reserved_blocks
        self.model_config = model_config
        self.device = device
        self.block_mgr = BlockManager_C(block_size, num_gpu_blocks, num_reserved_blocks)
        self.decode_seq_lens = {}  # Track sequence lengths for each request
    
    def reset_state(self):
        """Reset all state for clean benchmarking"""
        self.decode_seq_lens = {}
        # Note: C++ block manager state is reset by creating new instance
        # This is handled in the benchmark functions
    
    def update_block_table(self, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata):
        """Update block table using CPU block manager - follows original implementation"""
        init_seq_ids = meta_py.seq_ids[:meta_py.num_prefill_seqs]
        decode_seq_ids = meta_py.seq_ids
        
        # If the first layer in this attention worker, update block table and decode_seq_lens
        if meta_py.layer_id == self.model_config.layer_ids[0]:
            # Allocate kv blocks for init seqs, update for all decoding seqs
            for i, seq_id in enumerate(init_seq_ids):
                self.decode_seq_lens[seq_id] = meta_py.init_prefill_lens[i]
            
            decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]
            
            # Update block table
            self.block_mgr.update_block_table(meta_c, decode_seq_lens)
            
            # Increment sequence lengths for all sequences
            for i, seq_id in enumerate(decode_seq_ids):
                decode_seq_lens[i] += 1
                self.decode_seq_lens[seq_id] += 1
        else:
            decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]
        
        return decode_seq_lens
    
    def prepare_block_table(self, meta_c: AttentionBatchMetadata_C, decode_seq_lens: List[int]):
        """Prepare block table using CPU block manager"""
        return self.block_mgr.prepare_block_table(meta_c, decode_seq_lens)
    
    def pack_flash_attn_metadata(self, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata, 
                                decode_seq_lens: List[int], dummy_cache: bool = False):
        """Pack FlashAttention metadata using CPU approach - follows original implementation"""
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens
        
        # 1. prepare block table
        if dummy_cache:
            # dummy_cache is True when _warmup_attn
            block_table_1d = torch.zeros(
                (num_tokens + num_seqs * self.model_config.max_seq_len // self.block_size, ), 
                dtype=torch.int32, device=self.device)
        else:
            block_table_1d = self.block_mgr.prepare_block_table(meta_c, decode_seq_lens)

        slot_mapping_cuda = block_table_1d[-num_tokens:].to(torch.int64)
        block_table_cuda = block_table_1d[:-num_tokens].view(num_tokens, -1)

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        from disagmoe_c import prepare_batch_infos
        batch_infos_cuda = prepare_batch_infos(meta_c, decode_seq_lens)

        seq_lens_cuda, context_lens_cuda, seq_start_loc_cuda = \
            torch.split(batch_infos_cuda, [num_seqs, num_seqs, num_seqs + 1], dim=0)
        query_start_loc = torch.arange(num_tokens + 1, dtype=torch.int32, device=self.device)
            
        seq_lens = decode_seq_lens
        max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
        
        return FlashAttentionMetadata(
            0,
            0,
            num_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            max_decode_query_len=1,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
            block_tables=block_table_cuda,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
        )

class GPUBlockManager:
    """GPU-based block manager for comparison"""
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, device: str = "cuda"):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        
        # Initialize GPU-based components
        self.req_to_indice = {}
        self.req_seq_lens = torch.empty(1000, dtype=torch.int32, device=device)  # Large enough buffer
        self.decode_seq_lens = {}
        
        # Initialize token allocator and req pool
        self.max_running_reqs = (cache_config.num_gpu_blocks - 1) // 100 + 1
        self.req_to_token_pool = ReqToTokenPool(
            self.max_running_reqs,
            model_config.max_seq_len,
            device,
        )
        
        # Initialize token allocator (simplified version)
        from disagmoe.executor.block_manager import MHATokenToKVPool
        self.kv_cache = MHATokenToKVPool(
            cache_config.num_gpu_blocks,
            cache_config.block_size,
            model_config.dtype,
            model_config.num_kv_heads,
            model_config.hidden_size // model_config.num_heads,
            1,  # num_layers
            device,
        )
        
        self.token_allocator = TokenToKVPoolAllocator(
            cache_config.num_gpu_blocks,
            model_config.dtype,
            device,
            self.kv_cache,
            need_sort=False,
        )
    
    def reset_state(self):
        """Reset all state for clean benchmarking"""
        self.decode_seq_lens = {}
        self.req_to_indice = {}
        self.req_seq_lens.zero_()  # Reset tensor to zeros
        self.req_to_token_pool.clear()  # Reset the token pool
        self.token_allocator.clear()  # Reset the token allocator
    
    def update_block_table(self, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata):
        """Update block table using GPU-based approach"""
        init_seq_ids = meta_py.seq_ids[:meta_py.num_prefill_seqs]
        running_seq_ids = meta_py.seq_ids[meta_py.num_prefill_seqs:]
        seq_ids = meta_py.seq_ids
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        
        if meta_py.layer_id == 0:  # First layer
            # Allocate prefill token slots for init seqs
            new_req_indices = self.req_to_token_pool.alloc(meta_py.num_prefill_seqs)
            for i, seq_id in enumerate(init_seq_ids):
                req_indice = new_req_indices[i]
                self.req_to_indice[seq_id] = req_indice
                prefill_kv_locs = self.token_allocator.alloc(meta_py.init_prefill_lens[i])
                self.req_to_token_pool.write((req_indice, slice(0, meta_py.init_prefill_lens[i])), prefill_kv_locs)
                self.decode_seq_lens[seq_id] = meta_py.init_prefill_lens[i]
                
            running_req_indices = [self.req_to_indice.get(seq_id) for seq_id in running_seq_ids]
            batch_req_indices = running_req_indices + new_req_indices
            batch_req_indices_tensor = torch.tensor(batch_req_indices, dtype=torch.int32, device=self.device)
            
            seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in seq_ids]
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
            
            for i, seq_id in enumerate(seq_ids):
                seq_lens[i] += 1
                self.decode_seq_lens[seq_id] += 1
                
            increment_locs = self.token_allocator.alloc(num_tokens)
            self.req_to_token_pool.write_loc(batch_req_indices_tensor, seq_lens_tensor, increment_locs.to(torch.int32))
            seq_lens_tensor = seq_lens_tensor + 1
            self.req_seq_lens[batch_req_indices_tensor] = seq_lens_tensor
        else:
            seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in seq_ids]
            batch_req_indices = [self.req_to_indice.get(seq_id) for seq_id in seq_ids]
            batch_req_indices_tensor = torch.tensor(batch_req_indices, dtype=torch.int32, device=self.device)
            seq_lens_tensor = self.req_seq_lens[batch_req_indices_tensor]
            
        meta_py.seq_lens = seq_lens
        meta_py.seq_lens_tensor = seq_lens_tensor
        meta_py.req_indices = batch_req_indices
        meta_py.req_indices_tensor = batch_req_indices_tensor
        
        return seq_lens

def prefill_cpu_update_block_table_iter(cpu_mgr: CPUBlockManager, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("CPU/update_block_table/prefill_iter"):
        t0 = time.time()
        cpu_mgr.update_block_table(meta_c, meta_py)
        torch.cuda.synchronize()
    return time.time() - t0

def prefill_gpu_update_block_table_iter(gpu_mgr: GPUBlockManager, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("GPU/update_block_table/prefill_iter"):
        t0 = time.time()
        gpu_mgr.update_block_table(meta_c, meta_py)
        torch.cuda.synchronize()
    return time.time() - t0

def decode_cpu_update_block_table_iter_layer1(cpu_mgr: CPUBlockManager, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("CPU/update_block_table/decode_layer1_iter"):
        t0 = time.time()
        cpu_mgr.update_block_table(meta_c, meta_py)
        torch.cuda.synchronize()
    return time.time() - t0

def decode_gpu_update_block_table_iter_layer1(gpu_mgr: GPUBlockManager, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("GPU/update_block_table/decode_layer1_iter"):
        t0 = time.time()
        gpu_mgr.update_block_table(meta_c, meta_py)
        torch.cuda.synchronize()
    return time.time() - t0

def decode_cpu_pack_flash_attn_iter(cpu_mgr: CPUBlockManager, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata, decode_seq_lens: List[int]) -> float:
    torch.cuda.synchronize()
    with torch_profiler.record_function("CPU/pack_flash_attn_metadata/decode_iter"):
        t0 = time.time()
        _ = cpu_mgr.pack_flash_attn_metadata(meta_c, meta_py, decode_seq_lens)
        torch.cuda.synchronize()
    return time.time() - t0

def decode_gpu_pack_flash_attn_iter(gpu_mgr: GPUBlockManager, meta_py: AttentionBatchMetadata) -> float:
    with torch_profiler.record_function("GPU/pack_flash_attn_metadata/decode_iter"):
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        seq_lens_cuda = meta_py.seq_lens_tensor
        context_lens_cuda = seq_lens_cuda - 1
        seq_start_loc_cuda = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), 
                                       torch.cumsum(seq_lens_cuda, dim=0)])
        query_start_loc = torch.arange(num_tokens + 1, dtype=torch.int32, device="cuda")
        seq_lens = meta_py.seq_lens
        max_decode_seq_len = max(seq_lens) if len(seq_lens) > 0 else 0
        req_indices = meta_py.req_indices_tensor
        block_table_cuda = gpu_mgr.req_to_token_pool.get_block_table(req_indices, max_decode_seq_len)
        slot_mapping_cuda = gpu_mgr.req_to_token_pool.get_latest_loc(req_indices, seq_lens_cuda).to(torch.int64)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_tokens,
            slot_mapping=slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=1,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            max_decode_query_len=1,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
            block_tables=block_table_cuda,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )
        torch.cuda.synchronize()
    return time.time() - t0

def prefill_cpu_setup_iter(cpu_mgr: CPUBlockManager):
    with torch_profiler.record_function("setup/prefill/cpu_reset_iter"):
        cpu_mgr.reset_state()
        cpu_mgr.block_mgr = BlockManager_C(cpu_mgr.block_size, cpu_mgr.num_gpu_blocks, cpu_mgr.num_reserved_blocks)

def prefill_gpu_setup_iter(gpu_mgr: GPUBlockManager):
    with torch_profiler.record_function("setup/prefill/gpu_reset_iter"):
        gpu_mgr.reset_state()

def decode_cpu_setup_layer1_iter(cpu_mgr: CPUBlockManager, batch_size: int, seq_len: int):
    with torch_profiler.record_function("setup/decode_layer1/cpu_reset_and_alloc_iter"):
        cpu_mgr.reset_state()
        cpu_mgr.block_mgr = BlockManager_C(cpu_mgr.block_size, cpu_mgr.num_gpu_blocks, cpu_mgr.num_reserved_blocks)
        for i in range(batch_size):
            cpu_mgr.decode_seq_lens[i] = seq_len
        for i in range(batch_size):
            cpu_mgr.block_mgr.allocate(i, seq_len)

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
        cpu_mgr.block_mgr = BlockManager_C(cpu_mgr.block_size, cpu_mgr.num_gpu_blocks, cpu_mgr.num_reserved_blocks)
        decode_seq_lens: List[int] = []
        for i in range(batch_size):
            cpu_mgr.decode_seq_lens[i] = seq_len
            cpu_mgr.block_mgr.allocate(i, seq_len)
            decode_seq_lens.append(seq_len)
    return decode_seq_lens

def pack_gpu_setup_once(gpu_mgr: GPUBlockManager, meta_py: AttentionBatchMetadata, batch_size: int, seq_len: int):
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
        meta_py.seq_lens = [seq_len] * batch_size
        meta_py.seq_lens_tensor = seq_lens_tensor
        meta_py.req_indices = list(new_req_indices)
        meta_py.req_indices_tensor = batch_req_indices_tensor

def create_test_metadata(batch_size: int, num_prefill_seqs: int, seq_len: int, layer_id: int = 0) -> Tuple[AttentionBatchMetadata_C, AttentionBatchMetadata]:
    """Create test metadata for benchmarking"""
    num_prefill_tokens = num_prefill_seqs
    num_decode_tokens = batch_size - num_prefill_seqs
    
    # Create Python metadata
    meta_py = AttentionBatchMetadata(
        layer_id=layer_id,
        shape=[batch_size, 1024],  # hidden_size
        dtype="bf16",
        num_prefill_seqs=num_prefill_seqs,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        seq_ids=list(range(batch_size)),
        init_prefill_lens=[seq_len] * num_prefill_seqs,
        expert_ids=[0] * batch_size,
        topk_weights=[1.0] * batch_size,
        attn_dp_ranks=[0] * batch_size,
    )
    
    # Create C++ metadata
    meta_c = AttentionBatchMetadata_C()
    meta_c.layer_id = layer_id
    meta_c.shape = [batch_size, 1024]
    meta_c.dtype = "bf16"
    meta_c.num_prefill_seqs = num_prefill_seqs
    meta_c.num_prefill_tokens = num_prefill_tokens
    meta_c.num_decode_tokens = num_decode_tokens
    meta_c.seq_ids = list(range(batch_size))
    meta_c.init_prefill_lens = [seq_len] * num_prefill_seqs
    meta_c.expert_ids = [0] * batch_size
    meta_c.topk_weights = [1.0] * batch_size
    meta_c.attn_dp_ranks = [0] * batch_size
    
    return meta_c, meta_py

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
        # Create prefill-only metadata
        meta_c, meta_py = create_test_metadata(batch_size, batch_size, seq_len, layer_id=0)
        
        # Warmup (not timed)
        warmup_iters = 2
        for _ in range(warmup_iters):
            prefill_cpu_setup_iter(cpu_mgr)
            torch.cuda.synchronize()
            cpu_mgr.update_block_table(meta_c, meta_py)
        for _ in range(warmup_iters):
            prefill_gpu_setup_iter(gpu_mgr)
            torch.cuda.synchronize()
            gpu_mgr.update_block_table(meta_c, meta_py)
        
        # CPU version
        cpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset state (not timed)
            prefill_cpu_setup_iter(cpu_mgr)
            
            cpu_total_time += prefill_cpu_update_block_table_iter(cpu_mgr, meta_c, meta_py)
        cpu_time = cpu_total_time / num_iterations * 1000
        
        # GPU version
        gpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset state (not timed)
            prefill_gpu_setup_iter(gpu_mgr)
            
            gpu_total_time += prefill_gpu_update_block_table_iter(gpu_mgr, meta_c, meta_py)
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
        # Create decode-only metadata and set to layer 1
        meta_c, meta_py = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
        meta_c.layer_id = 1
        meta_py.layer_id = 1
        
        # Warmup (not timed)
        warmup_iters = 2
        for _ in range(warmup_iters):
            decode_cpu_setup_layer1_iter(cpu_mgr, batch_size, seq_len)
            torch.cuda.synchronize()
            cpu_mgr.update_block_table(meta_c, meta_py)
        for _ in range(warmup_iters):
            decode_gpu_setup_layer1_iter(gpu_mgr, batch_size, seq_len)
            torch.cuda.synchronize()
            gpu_mgr.update_block_table(meta_c, meta_py)

        # CPU version
        cpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset and prepare state (not timed)
            decode_cpu_setup_layer1_iter(cpu_mgr, batch_size, seq_len)
            
            cpu_total_time += decode_cpu_update_block_table_iter_layer1(cpu_mgr, meta_c, meta_py)
        cpu_time = cpu_total_time / num_iterations * 1000
        
        # GPU version
        gpu_total_time = 0.0
        for _ in range(num_iterations):
            # Reset and prepare state (not timed)
            decode_gpu_setup_layer1_iter(gpu_mgr, batch_size, seq_len)
            
            gpu_total_time += decode_gpu_update_block_table_iter_layer1(gpu_mgr, meta_c, meta_py)
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
    print(f"{'Batch Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-"*50)
    
    for batch_size in batch_sizes:
        # Create decode-only metadata at layer 1
        meta_c, meta_py = create_test_metadata(batch_size, 0, seq_len, layer_id=1)
        meta_c.layer_id = 1
        meta_py.layer_id = 1
        
        # CPU version - benchmark with proper cleanup between iterations
        cpu_total_time = 0.0
        # Warmup (not timed)
        warmup_iters = 2
        for _ in range(warmup_iters):
            decode_seq_lens = pack_cpu_setup_iter(cpu_mgr, batch_size, seq_len)
            torch.cuda.synchronize()
            _ = cpu_mgr.pack_flash_attn_metadata(meta_c, meta_py, decode_seq_lens)
        for _ in range(num_iterations):
            # Reset state and allocate (not timed)
            with torch_profiler.record_function("setup/pack_metadata/cpu_reset_and_alloc"):
                cpu_mgr.reset_state()
                cpu_mgr.block_mgr = BlockManager_C(cpu_mgr.block_size, cpu_mgr.num_gpu_blocks, cpu_mgr.num_reserved_blocks)
                # Pre-populate decode_seq_lens and allocate block tables (not timed)
                decode_seq_lens = []
                for i in range(batch_size):
                    cpu_mgr.decode_seq_lens[i] = seq_len
                    cpu_mgr.block_mgr.allocate(i, seq_len)
                    decode_seq_lens.append(seq_len)
            
            # Pack flash attention metadata (timed)
            cpu_total_time += decode_cpu_pack_flash_attn_iter(cpu_mgr, meta_c, meta_py, decode_seq_lens)
        cpu_time = cpu_total_time / num_iterations * 1000
        
        # GPU version - setup once, then benchmark only metadata packing
        # Reset state and setup for decode-only (not timed)
        pack_gpu_setup_once(gpu_mgr, meta_py, batch_size, seq_len)
        # Warmup (not timed)
        warmup_iters = 2
        for _ in range(warmup_iters):
            num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
            seq_lens_cuda = meta_py.seq_lens_tensor
            context_lens_cuda = seq_lens_cuda - 1
            seq_start_loc_cuda = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), 
                                           torch.cumsum(seq_lens_cuda, dim=0)])
            query_start_loc = torch.arange(num_tokens + 1, dtype=torch.int32, device="cuda")
            req_indices = meta_py.req_indices_tensor
            max_decode_seq_len = max(meta_py.seq_lens) if len(meta_py.seq_lens) > 0 else 0
            block_table_cuda = gpu_mgr.req_to_token_pool.get_block_table(req_indices, max_decode_seq_len)
            slot_mapping_cuda = gpu_mgr.req_to_token_pool.get_latest_loc(req_indices, seq_lens_cuda).to(torch.int64)
            _ = FlashAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_tokens,
                slot_mapping=slot_mapping_cuda,
                seq_lens=meta_py.seq_lens,
                seq_lens_tensor=seq_lens_cuda,
                max_query_len=1,
                max_prefill_seq_len=0,
                max_decode_seq_len=max_decode_seq_len,
                max_decode_query_len=1,
                query_start_loc=query_start_loc,
                seq_start_loc=seq_start_loc_cuda,
                context_lens_tensor=context_lens_cuda,
                block_tables=block_table_cuda,
                use_cuda_graph=False,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=False,
            )
        
        # Now benchmark only the metadata packing
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            # Pack flash attention metadata using GPU approach (timed)
            _ = decode_gpu_pack_flash_attn_iter(gpu_mgr, meta_py)
        
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / num_iterations * 1000
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results['cpu_times'].append(cpu_time)
        results['gpu_times'].append(gpu_time)
        results['speedups'].append(speedup)
        
        print(f"{batch_size:<12} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<10.2f}")
    
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
    
    cache_config = CacheConfig(
        block_size=1,
        gpu_memory_utilization=0.8,
        swap_space=0,
        cache_dtype="auto",
        num_gpu_blocks=80000,
        num_reserved_blocks=1024,
    )
    
    # Initialize managers
    cpu_mgr = CPUBlockManager(cache_config.block_size, cache_config.num_gpu_blocks, cache_config.num_reserved_blocks, model_config)
    gpu_mgr = GPUBlockManager(model_config, cache_config)
    
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
    
    profiler = None
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