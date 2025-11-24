"""
Benchmark script for cuda_graph_preprocess kernel.
Tests both fused_copy (Triton kernel) and non_fused_copy (PyTorch copy) modes.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from argparse import ArgumentParser

from torch.profiler import ProfilerActivity
from disagmoe.ops.cuda_graph import cuda_graph_preprocess_triton, cuda_graph_preprocess_triton_opt, cuda_graph_preprocess_cuda
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

class CudaGraphPreprocessBenchmark:
    """Benchmark class for cuda_graph_preprocess operations."""
    
    def __init__(self, max_batch_size: int, hidden_size: int, max_seq_len: int, block_size: int):
        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.max_num_blocks = (max_seq_len + block_size - 1) // block_size
        
        # Create static buffers (similar to CUDAGraphAttnExecutor)
        self.static_input = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
        self.static_positions = torch.zeros(max_batch_size, dtype=torch.long, device="cuda")
        self.static_block_table = torch.zeros(
            (max_batch_size, self.max_num_blocks), 
            dtype=torch.int32, device="cuda")
        self.static_slot_mapping = torch.zeros((max_batch_size,), dtype=torch.long, device="cuda")
        
        self.static_batch_info = torch.zeros(
            (max_batch_size + max_batch_size + (max_batch_size + 1) + (max_batch_size + 1)), 
            dtype=torch.int32, device="cuda")
        static_batch_info_splits = self.static_batch_info.split(
            [max_batch_size, max_batch_size, max_batch_size + 1, max_batch_size + 1])
        self.static_seq_lens = static_batch_info_splits[0]
        self.static_context_lens = static_batch_info_splits[1]
        self.static_seq_start_loc = static_batch_info_splits[2]
        self.static_query_start_loc = static_batch_info_splits[3]
        self.static_query_start_loc.copy_(torch.arange(max_batch_size + 1, dtype=torch.int32, device="cuda"))
        
        self.func_map = {
            "fused": self.cuda_graph_preprocess_fused,
            "fused_new": self.cuda_graph_preprocess_fused_new,
            "fused_cuda": self.cuda_graph_preprocess_fused_cuda,
            "non_fused": self.cuda_graph_preprocess_non_fused,
        }
    
    def create_test_data(self, num_tokens: int, hidden_dim: int, max_num_blocks: int) -> Tuple[torch.Tensor, torch.Tensor, FlashAttentionMetadata]:
        """Create test data for benchmarking."""
        # Create random input data
        hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
        positions = torch.randint(0, self.max_seq_len, (num_tokens,), dtype=torch.long, device="cuda")
        
        # Create metadata
        seq_lens = torch.ones(num_tokens, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")
        
        # Create block tables
        block_tables = torch.zeros((num_tokens, max_num_blocks), dtype=torch.int32, device="cuda")
        for i in range(num_tokens):
            num_blocks = min(max_num_blocks, (i + 1) // self.block_size + 1)
            block_tables[i, :num_blocks] = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
        
        # Create seq_start_loc
        seq_start_loc = torch.arange(num_tokens + 1, dtype=torch.int32, device="cuda")
        
        meta = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens.tolist(),
            seq_lens_tensor=seq_lens,
            max_query_len=1,
            max_prefill_seq_len=0,
            max_decode_seq_len=1,
            max_decode_query_len=1,
            query_start_loc=None,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens,
            block_tables=block_tables,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
        )
        
        return hidden_states, positions, meta
    
    def cuda_graph_preprocess_fused(self, hidden_states: torch.Tensor, positions: torch.Tensor, meta: FlashAttentionMetadata):
        """Fused copy using Triton kernel."""
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        max_num_blocks = meta.block_tables.shape[1]
        
        TOKEN_BLOCK_SIZE = 64
        HIDDEN_BLOCK_SIZE = 256
        grid_token = (num_tokens + TOKEN_BLOCK_SIZE - 1) // TOKEN_BLOCK_SIZE
        grid_hidden = (hidden_dim + HIDDEN_BLOCK_SIZE - 1) // HIDDEN_BLOCK_SIZE
        grid_blocks = (max_num_blocks + HIDDEN_BLOCK_SIZE - 1) // HIDDEN_BLOCK_SIZE
        
        cuda_graph_preprocess_triton[(grid_token, max(grid_hidden, grid_blocks))](
            # Destination pointers
            self.static_input, self.static_positions, self.static_slot_mapping,
            self.static_block_table, self.static_seq_lens, self.static_context_lens,
            self.static_seq_start_loc,
            # Source pointers
            hidden_states, positions, meta.slot_mapping,
            meta.block_tables, meta.seq_lens_tensor, meta.context_lens_tensor,
            meta.seq_start_loc,
            # Dimensions
            num_tokens, hidden_dim, max_num_blocks,
            # Strides
            self.static_input.stride(0), hidden_states.stride(0),
            self.static_block_table.stride(0), self.static_block_table.stride(1),
            meta.block_tables.stride(0), meta.block_tables.stride(1),
            TOKEN_BLOCK_SIZE=TOKEN_BLOCK_SIZE, HIDDEN_BLOCK_SIZE=HIDDEN_BLOCK_SIZE,
        )
        
    def cuda_graph_preprocess_fused_new(self, hidden_states: torch.Tensor, positions: torch.Tensor, meta: FlashAttentionMetadata):
        T, H = hidden_states.shape
        B = meta.block_tables.shape[1]

        BLOCK = 256
        grid = (T + 1,)   # T per-token blocks + 1 control block

        cuda_graph_preprocess_triton_opt[grid](
            hidden_states, meta.block_tables,
            positions, meta.slot_mapping,
            meta.seq_lens_tensor, meta.context_lens_tensor,
            meta.seq_start_loc,

            self.static_input,
            self.static_block_table,
            self.static_positions,
            self.static_slot_mapping,
            self.static_seq_lens,
            self.static_context_lens,
            self.static_seq_start_loc,

            T=T, H=H, B=B, BLOCK=BLOCK
        )
        
    def cuda_graph_preprocess_fused_cuda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        meta: FlashAttentionMetadata,
        tokens_per_block=4
    ):
        cuda_graph_preprocess_cuda(
            hidden_states,
            positions,
            meta.block_tables,
            meta.slot_mapping,
            meta.seq_lens_tensor,
            meta.context_lens_tensor,
            meta.seq_start_loc,

            self.static_input,
            self.static_positions,
            self.static_block_table,
            self.static_slot_mapping,
            self.static_seq_lens,
            self.static_context_lens,
            self.static_seq_start_loc,

            tokens_per_block=tokens_per_block
        )
    
    def cuda_graph_preprocess_non_fused(self, hidden_states: torch.Tensor, positions: torch.Tensor, meta: FlashAttentionMetadata):
        """Non-fused copy using PyTorch operations."""
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        max_num_blocks = meta.block_tables.shape[1]
        
        self.static_input[:num_tokens, :hidden_dim].copy_(hidden_states)
        self.static_positions[:num_tokens].copy_(positions)
        self.static_slot_mapping[:num_tokens].copy_(meta.slot_mapping)
        self.static_block_table[:num_tokens, :max_num_blocks].copy_(meta.block_tables)
        self.static_seq_lens[:num_tokens].copy_(meta.seq_lens_tensor)
        self.static_context_lens[:num_tokens].copy_(meta.context_lens_tensor)
        self.static_seq_start_loc[:num_tokens + 1].copy_(meta.seq_start_loc)
    
    def benchmark_single(self, num_tokens: int, hidden_dim: int, max_num_blocks: int, 
                        method: str, num_warmup: int = 10, num_iterations: int = 20) -> float:
        """Benchmark a single configuration."""
        hidden_states, positions, meta = self.create_test_data(num_tokens, hidden_dim, max_num_blocks)
        
        # Warmup
        for _ in range(num_warmup):
            self.func_map[method](hidden_states, positions, meta)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iterations):
            self.func_map[method](hidden_states, positions, meta)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_iterations
        
        return avg_time_ms
    
    def benchmark_configs(self, configs: List[Tuple[int, int, int]], 
                         num_warmup: int = 10, num_iterations: int = 20):
        """Benchmark multiple configurations."""
        method_names = list(self.func_map.keys())
        print(f"{'Config':<30}" + " ".join([f'{method_names[i]:<15} (us)' for i in range(len(method_names))]))
        print("-" * 75)
        
        for num_tokens, hidden_dim, max_num_blocks in configs:
            print(f'({num_tokens:4d},{hidden_dim:4d},{max_num_blocks:4d})', end=' | ')
            for method in method_names: 
                time = self.benchmark_single(num_tokens, hidden_dim, max_num_blocks, 
                                              method=method, num_warmup=num_warmup, 
                                              num_iterations=num_iterations)
                print(f'{time*1000:<15.2f}', end=' | ')
            print()


def main():
    parser = ArgumentParser(description="Benchmark cuda_graph_preprocess kernel")
    parser.add_argument("--max-batch-size", type=int, default=160, help="Maximum batch size")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden dimension size")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--block-size", type=int, default=16, help="Block size for KV cache")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--config-file", type=str, default=None, 
                       help="Path to config file with test configurations (one per line: num_tokens,hidden_dim,max_num_blocks)")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = CudaGraphPreprocessBenchmark(
        max_batch_size=args.max_batch_size,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        block_size=args.block_size,
    )
    
    # Define test configurations
    if args.config_file:
        # Load from file
        configs = []
        with open(args.config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) == 3:
                        configs.append((int(parts[0]), int(parts[1]), int(parts[2])))
    else:
        # Default configurations
        configs = [
            # (num_tokens, hidden_dim, max_num_blocks)
            (1, 4096, 256),
            (8, 4096, 256),
            (16, 4096, 256),
            (32, 4096, 256),
            (64, 4096, 256),
            (128, 4096, 256),
            (160, 4096, 256),
            (32, 2048, 128),
            (64, 2048, 128),
            (128, 2048, 128),
        ]
    
    print("=" * 75)
    print("CUDA Graph Preprocess Benchmark")
    print("=" * 75)
    print(f"Max batch size: {args.max_batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Block size: {args.block_size}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iterations}")
    print("=" * 75)
    print()
    
    # Run benchmarks
    # profiler = torch.profiler.profile(
    #     record_shapes=True,
    #     with_stack=True,
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # )
    # profiler.start()
    benchmark.benchmark_configs(
        configs, 
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations
    )
    # profiler.stop()
    # profiler.export_chrome_trace(f"cuda_graph_preprocess.trace")

if __name__ == "__main__":
    main()

