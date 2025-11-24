import torch

from torch import Tensor
import numpy as np

from typing import Tuple, List, Dict
import time

from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from disagmoe.config import ModelConfig, CacheConfig as DmoeCacheConfig
from disagmoe.utils.logger import get_logger
from disagmoe.models.utils import make_attention_dummy_batch
from disagmoe.ops.cuda_graph import cuda_graph_preprocess_cuda

class CUDAGraphAttnExecutor:
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig, attn_executor):
        self.model_config = model_config
        self.cache_config = cache_config
        self.attn_executor = attn_executor
        self.fused_copy = True
        
    def create_cuda_graph_buffers(self):
        assert self.model_config.enable_cuda_graph_attn
        batch_size = self.model_config.max_batch_size_attn
        self.graphs: Dict[int, List[torch.cuda.CUDAGraph]] = {}
        self.static_outputs: Dict[int, List[Tuple[Tensor]]] = {}

        self.static_input = torch.zeros((batch_size, self.model_config.hidden_size), dtype=self.model_config.dtype, device="cuda")
        self.static_positions = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        self.static_block_table = torch.zeros(
            (batch_size, self.model_config.max_seq_len // self.cache_config.block_size), 
            dtype=torch.int32, device="cuda")
        self.static_slot_mapping = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")

        self.static_batch_info = torch.zeros((batch_size + batch_size + (batch_size + 1) + (batch_size + 1)), dtype=torch.int32, device="cuda")
        static_batch_info_splits = self.static_batch_info.split([batch_size, batch_size, batch_size + 1, batch_size + 1])
        self.static_seq_lens = static_batch_info_splits[0]
        self.static_context_lens = static_batch_info_splits[1]
        self.static_seq_start_loc = static_batch_info_splits[2]
        self.static_query_start_loc = static_batch_info_splits[3]
        self.static_query_start_loc.copy_(torch.arange(batch_size + 1, dtype=torch.int32, device="cuda"))

        self.static_batch_infos: Dict[int, Tensor] = {}

        self.graph_batch_sizes = list(range(max(self.model_config.graph_stride, self.model_config.ep_size),
                                            batch_size + 1,
                                            self.model_config.graph_stride))
        self.graph_batch_sizes = [1] + self.graph_batch_sizes

        for layer_id in self.model_config.layer_ids:
            self.graphs[layer_id] = [torch.cuda.CUDAGraph() for _ in self.graph_batch_sizes]
            self.static_outputs[layer_id] = []
        
        for bs in self.graph_batch_sizes:
            self.static_batch_infos[bs] = torch.zeros((bs + bs + (bs + 1) + (bs + 1)), dtype=torch.int32, device="cuda")
            
    def prepare_metadata_for_capture(self, meta: FlashAttentionMetadata):
        num_tokens = meta.num_prefill_tokens + meta.num_decode_tokens
        max_num_blocks = meta.block_tables.shape[1]
        return FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_tokens,
            slot_mapping=self.static_slot_mapping[ : num_tokens],
            seq_lens=meta.seq_lens,
            seq_lens_tensor=self.static_seq_lens[ : num_tokens],
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=meta.max_decode_seq_len,
            max_decode_query_len=1,
            query_start_loc=self.static_query_start_loc[ : num_tokens + 1],
            seq_start_loc=self.static_seq_start_loc[ : num_tokens + 1],
            context_lens_tensor=self.static_context_lens[ : num_tokens],
            block_tables=self.static_block_table[ : num_tokens, : max_num_blocks],
            use_cuda_graph=meta.use_cuda_graph,
            multi_modal_placeholder_index_maps=meta.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=meta.enable_kv_scales_calculation,
        )
        
    def capture(self):
        start_time = time.perf_counter()
        get_logger().info(f"Capturing CUDA graphs for attention, bsz {self.graph_batch_sizes}")
        for layer_id in self.model_config.layer_ids:
            for graph, graph_batch_size in zip(self.graphs[layer_id], self.graph_batch_sizes):
                batch = make_attention_dummy_batch(0, graph_batch_size, self.model_config.hidden_size, self.model_config.max_seq_len)
                attn_meta = self.attn_executor.block_mgr.pack_flash_attn_metadata(batch.to_metadata_c(), batch, dummy_cache=True)
                self.cuda_graph_preprocess(batch.data, batch.seq_lens_tensor.to(torch.long), attn_meta)
                graph_attn_meta = self.prepare_metadata_for_capture(attn_meta)

                def run_once() -> Tuple[Tensor, Tensor, Tensor]:
                    # Provide dummy request IDs from the synthetic batch to satisfy profile-driven gating.
                    return self.attn_executor.execute_eager(
                        layer_id, self.static_positions[ : graph_batch_size], 
                        self.static_input[ : graph_batch_size], graph_attn_meta,
                        request_ids=batch.req_ids
                    )

                for _ in range(2):
                    # warmup
                    run_once()

                with torch.cuda.graph(graph):
                    outputs = run_once()
                    
                self.static_outputs[layer_id].append(outputs)
                
        end_time = time.perf_counter()
        torch.cuda.synchronize()
        get_logger().info(f"cuda graph captured in {end_time - start_time} seconds")
        self.test_graph()
        
    def test_graph(self):
        start_time = time.perf_counter()
        for layer_id in self.model_config.layer_ids:
            for bs in self.graph_batch_sizes:
                batch = make_attention_dummy_batch(0, bs, self.model_config.hidden_size, self.model_config.max_seq_len)
                meta = self.attn_executor.block_mgr.pack_flash_attn_metadata(batch.to_metadata_c(), batch, dummy_cache=True)
                hiddens, expert_weights, expert_ids = self.run(layer_id, batch.seq_lens_tensor.to(torch.long), batch.data, meta)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        get_logger().info(f"cuda graph test passed in {end_time - start_time} seconds")

    def _get_graph_by_batch_size(self, batch_size: int):
        for i, size in enumerate(self.graph_batch_sizes):
            if size >= batch_size:
                return i, size
        assert False, f"No available graph for batch size={batch_size}"
        
    def cuda_graph_preprocess(self, hidden_states: torch.Tensor, positions: torch.Tensor, meta: FlashAttentionMetadata):
        num_tokens = hidden_states.shape[0]
        max_num_blocks = meta.block_tables.shape[1]
        
        if self.fused_copy:
            # skip query start loc copy as its value is pre-assigned
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
            )
        else:
            self.static_input[ : num_tokens].copy_(hidden_states)
            self.static_positions[ : num_tokens].copy_(positions)
            self.static_slot_mapping[ : num_tokens].copy_(meta.slot_mapping)
            self.static_block_table[ : num_tokens, : max_num_blocks].copy_(meta.block_tables)
            self.static_seq_lens[ : num_tokens].copy_(meta.seq_lens_tensor)
            self.static_context_lens[ : num_tokens].copy_(meta.context_lens_tensor)
            self.static_seq_start_loc[ : num_tokens + 1].copy_(meta.seq_start_loc)

    def run(self, layer_id: int, positions: torch.Tensor, hidden_states: torch.Tensor, meta: FlashAttentionMetadata) -> Tuple[Tensor, Tensor, Tensor]:
        meta.use_cuda_graph = True
        
        num_tokens = hidden_states.shape[0]
        graph_id, batch_size = self._get_graph_by_batch_size(num_tokens)
        
        self.cuda_graph_preprocess(hidden_states, positions, meta)
        self.graphs[layer_id][graph_id].replay()

        outputs, topk_weights, topk_ids = self.static_outputs[layer_id][graph_id]

        return outputs[ : num_tokens], topk_weights[ : num_tokens], topk_ids[ : num_tokens]
        