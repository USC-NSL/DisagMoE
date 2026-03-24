from disagmoe.frontend.datatypes import ExpertForwardBatch
import torch

from torch import Tensor
import numpy as np

from typing import Tuple, List, Dict
import time

from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from disagmoe.config import ModelConfig, CacheConfig as DmoeCacheConfig
from disagmoe.utils.logger import get_logger
from disagmoe.models.utils import make_attention_dummy_batch, make_expert_dummy_inputs
from disagmoe.models.experts import MoEExpertsCUTLASS, MoEExpertsCUTLASSFP8
from disagmoe.ops.cuda_graph import cuda_graph_preprocess_cuda, fused_copy_and_pad_cuda
from disagmoe.frontend.engine_utils import get_global_engine_config
from disagmoe.utils.tensor_utils import (
    get_cuda_aligned_tensor,
    make_tensor_view,
    bind_tensor_view_1d,
    bind_tensor_view_2d,
)

STATIC_BUFFER_ALIGNMENT = 16 # float4/int4 alignment

class CUDAGraphAttnExecutor:
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig, attn_executor):
        self.model_config = model_config
        self.cache_config = cache_config
        self.attn_executor = attn_executor
        self.fused_copy = True
        self.has_profile_gating = getattr(attn_executor, 'gate_profile_bytes', None) is not None \
            and len(attn_executor.gate_profile_bytes) > 0
        
    def create_cuda_graph_buffers(self):
        assert get_global_engine_config().enable_cuda_graph_attn
        batch_size = get_global_engine_config().max_attn_graph_bsz
        self.graphs: Dict[int, List[torch.cuda.CUDAGraph]] = {} # batch size -> graph list (one graph for each layer)
        self.static_outputs: Dict[int, List[Tuple[Tensor]]] = {} # batch size -> output list (one output for each layer)

        self.static_input = get_cuda_aligned_tensor(batch_size * self.model_config.hidden_size, self.model_config.dtype, alignment=STATIC_BUFFER_ALIGNMENT, device="cuda")
        self.static_input = self.static_input.reshape(batch_size, self.model_config.hidden_size)
        self.static_block_table = get_cuda_aligned_tensor(batch_size * self.model_config.max_seq_len // self.cache_config.block_size, torch.int32, alignment=STATIC_BUFFER_ALIGNMENT, device="cuda")
        self.static_block_table = self.static_block_table.reshape(batch_size, self.model_config.max_seq_len // self.cache_config.block_size)

        self.static_slot_mapping = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")
        
        self.static_positions = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        if self.has_profile_gating:
            self.static_request_ids = torch.zeros(batch_size, dtype=torch.int64, device="cuda")

        self.static_batch_info = torch.zeros((batch_size + batch_size + (batch_size + 1) + (batch_size + 1)), dtype=torch.int32, device="cuda")
        static_batch_info_splits = self.static_batch_info.split([batch_size, batch_size, batch_size + 1, batch_size + 1])
        self.static_seq_lens = static_batch_info_splits[0]
        self.static_context_lens = static_batch_info_splits[1]
        self.static_seq_start_loc = static_batch_info_splits[2]
        self.static_query_start_loc = static_batch_info_splits[3]
        self.static_query_start_loc.copy_(torch.arange(batch_size + 1, dtype=torch.int32, device="cuda"))

        self.static_batch_infos: Dict[int, Tensor] = {}

        self.graph_batch_sizes = self.get_graph_batch_sizes(batch_size)

        for bs in self.graph_batch_sizes:
            self.graphs[bs] = [torch.cuda.CUDAGraph() for _ in self.model_config.layer_ids]
            self.static_outputs[bs] = []
            self.static_batch_infos[bs] = torch.zeros((bs + bs + (bs + 1) + (bs + 1)), dtype=torch.int32, device="cuda")
            
        self.output_view = make_tensor_view(dtype=self.model_config.dtype)
        self.topk_ids_view = make_tensor_view(dtype=torch.int32)
        self.topk_weights_view = make_tensor_view(dtype=torch.float32)
            
    def get_graph_batch_sizes(self, graph_max_batch_size: int):
        assert graph_max_batch_size <= 1024
        graph_bsz = [1]
        bsz_stage = [8, 128, 256, 512, 1024]
        bsz_inc = [0, 32, 64, 128, 256]
        
        for i in range(1, len(bsz_stage)):
            if graph_max_batch_size > bsz_stage[i]:
                graph_bsz.extend(list(range(bsz_stage[i-1], bsz_stage[i], bsz_inc[i])))
            else:
                graph_bsz.extend(list(range(bsz_stage[i-1], graph_max_batch_size, bsz_inc[i])))
                if graph_bsz[-1] != graph_max_batch_size:
                    graph_bsz.append(graph_max_batch_size)
                break
        return graph_bsz
            
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
        layer_time_elapse = []
        layer_memory_elapse = []
        get_logger().info(f"Capturing CUDA graphs for attention, bsz {self.graph_batch_sizes}")
        
        for graph_batch_size in self.graph_batch_sizes:
            graph_list = self.graphs[graph_batch_size]
            batch = make_attention_dummy_batch(0, graph_batch_size, self.model_config.hidden_size, self.model_config.max_seq_len)
            attn_meta = self.attn_executor.block_mgr.pack_flash_attn_metadata(batch.to_metadata_c(), batch, dummy_cache=True)
            self.cuda_graph_preprocess(batch.data, batch.seq_lens_tensor.to(torch.long), attn_meta, graph_batch_size)
            graph_attn_meta = self.prepare_metadata_for_capture(attn_meta)
            bsz_start_time = time.perf_counter()
            free_memory_before, _ = torch.cuda.mem_get_info()
            if self.has_profile_gating:
                self.static_request_ids[ : graph_batch_size].copy_(
                    torch.arange(graph_batch_size, dtype=torch.int64, device="cuda")
                )
            for layer_id in self.model_config.layer_ids:
                graph = graph_list[layer_id]

                def run_once() -> Tuple[Tensor, Tensor, Tensor]:
                    req_ids = self.static_request_ids[ : graph_batch_size] if self.has_profile_gating else None
                    return self.attn_executor.execute_eager(
                        layer_id, self.static_positions[ : graph_batch_size], 
                        self.static_input[ : graph_batch_size], graph_attn_meta,
                        request_ids=req_ids
                    )

                # warmup
                for _ in range(2):
                    run_once()
                    
                time_before_capture = time.perf_counter()
                    
                if layer_id == 0:
                    with torch.cuda.graph(graph):
                        outputs = run_once()
                else:
                    with torch.cuda.graph(graph, pool=graph_list[0].pool()):
                        outputs = run_once()
                        
                time_after_capture = time.perf_counter()
                if layer_id == 0 and graph_batch_size > 1:
                    get_logger().info(f"Time taken to capture graph: {time_after_capture - time_before_capture} seconds")
                    
                self.static_outputs[graph_batch_size].append(outputs)
                
            bsz_end_time = time.perf_counter()
            layer_time_elapse.append(bsz_end_time - bsz_start_time)
            free_memory_after, _ = torch.cuda.mem_get_info()
            layer_memory_elapse.append(round((free_memory_before - free_memory_after) / (1024 ** 3), 2))
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        get_logger().info(f"Attention cuda graphs captured in {end_time - start_time} seconds.")
        # get_logger().info(f"Layer time elapse: {layer_time_elapse}")
        # get_logger().info(f"Layer memory elapse: {layer_memory_elapse}")
        self.test_graph()

    def test_graph(self):
        for layer_id in self.model_config.layer_ids:
            for bs in self.graph_batch_sizes:
                batch = make_attention_dummy_batch(0, bs, self.model_config.hidden_size, self.model_config.max_seq_len)
                meta = self.attn_executor.block_mgr.pack_flash_attn_metadata(batch.to_metadata_c(), batch, dummy_cache=True)
                request_ids = torch.arange(bs, dtype=torch.int64, device="cuda") if self.has_profile_gating else None
                hiddens, expert_weights, expert_ids = self.run(layer_id, batch.seq_lens_tensor.to(torch.long), batch.data, meta, request_ids=request_ids)
        torch.cuda.synchronize()

    def _get_bucket_by_num_tokens(self, batch_size: int):
        for size in self.graph_batch_sizes:
            if size >= batch_size:
                return size
        assert False, f"No available graph for batch size={batch_size}"
        
    def cuda_graph_preprocess(self, hidden_states: torch.Tensor, positions: torch.Tensor, meta: FlashAttentionMetadata, padded_batch_size: int):
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
                padded_batch_size,
            )
        else:
            self.static_input[ : num_tokens].copy_(hidden_states)
            self.static_positions[ : num_tokens].copy_(positions)
            self.static_block_table[ : num_tokens, : max_num_blocks].copy_(meta.block_tables)
            self.static_slot_mapping[ : num_tokens].copy_(meta.slot_mapping)
            self.static_seq_lens[ : num_tokens].copy_(meta.seq_lens_tensor)
            self.static_context_lens[ : num_tokens].copy_(meta.context_lens_tensor)
            self.static_seq_start_loc[ : num_tokens + 1].copy_(meta.seq_start_loc)

    def run(self, layer_id: int, positions: torch.Tensor, hidden_states: torch.Tensor, meta: FlashAttentionMetadata, request_ids: torch.Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        meta.use_cuda_graph = True
        
        try:
            num_tokens = hidden_states.shape[0]
            batch_size = self._get_bucket_by_num_tokens(num_tokens)
            
            self.cuda_graph_preprocess(hidden_states, positions, meta, batch_size)
            if self.has_profile_gating and request_ids is not None:
                self.static_request_ids[ : num_tokens].copy_(request_ids)
            self.graphs[batch_size][layer_id].replay()

            outputs, topk_weights, topk_ids = self.static_outputs[batch_size][layer_id]
            
        except Exception as e:
            get_logger().error(f"Error in run: {e}, layer_id {layer_id}, batch_size {num_tokens}, graph_bsz {batch_size}")
            raise e
        
        hidden_size = self.model_config.hidden_size
        topk = self.model_config.top_k
        
        use_view = False
        if use_view:
            bind_tensor_view_2d(self.output_view, outputs, 0, num_tokens, hidden_size, hidden_size)
            bind_tensor_view_2d(self.topk_ids_view, topk_ids, 0, num_tokens, topk, topk)
            bind_tensor_view_2d(self.topk_weights_view, topk_weights, 0, num_tokens, topk, topk)
            
            return self.output_view, self.topk_weights_view, self.topk_ids_view
        
        else:
            return outputs[ : num_tokens], topk_weights[ : num_tokens], topk_ids[ : num_tokens]

class CUDAGraphExpertsExecutor:

    def __init__(self, model_config: ModelConfig, local_to_global_expert_rank: List[int], global_to_local_expert_rank: List[int], experts_executor):
        self.model_config = model_config
        self.local_num_experts = len(local_to_global_expert_rank)
        self.local_to_global_expert_rank = local_to_global_expert_rank
        self.global_to_local_expert_rank = global_to_local_expert_rank
        self.experts_executor = experts_executor
        self.expert_cls = experts_executor.expert_cls

        self.expert_ids = torch.arange(self.local_num_experts, device="cpu", dtype=torch.int32)
        
    def create_cuda_graph_buffers(self):
        # BF16 activations in the graph; MoEExpertsCUTLASSFP8 quantizes inside forward.
        assert self.model_config.dtype == torch.bfloat16

        max_batch_size = get_global_engine_config().max_batch_size_expert
        self.static_outputs: Dict[int, List[Tensor]] = {} # callee allocated, no need to pre-allocate
        
        # Allocate respecting max batch size, small batches can use slices of this
        self.static_input_batch_sizes = torch.empty((self.local_num_experts,), dtype=torch.int64, device="cuda")
        self.static_input_hiddens = torch.empty((max_batch_size, self.model_config.hidden_size), dtype=self.model_config.dtype, device="cuda")
        self.static_input_m_indices = torch.empty((max_batch_size,), dtype=torch.int32, device="cuda")

        self.graph_batch_sizes = self.get_graph_batch_sizes(max_batch_size)
        self.graphs: Dict[int, List[torch.cuda.CUDAGraph]] = {}

        # initialize graphs
        for bs in self.graph_batch_sizes:
            self.graphs[bs] = [torch.cuda.CUDAGraph() for _ in self.model_config.layer_ids] # graphs of all layers
            self.static_outputs[bs] = []
        
    def get_graph_batch_sizes(self, graph_max_batch_size: int) -> List[int]:
        assert graph_max_batch_size > 0

        if graph_max_batch_size <= 128:
            return [graph_max_batch_size]
        
        graph_bsz: List[int] = []
        bsz = 128
        while bsz < graph_max_batch_size:
            graph_bsz.append(bsz)
            bsz *= 2
        graph_bsz.append(bsz) # always include the max

        return graph_bsz
    
    def cuda_graph_preprocess(self, hidden_states: torch.Tensor, batch_sizes: torch.Tensor, m_indices: torch.Tensor, bucket_size: int):
        num_tokens = hidden_states.shape[0]
        if self.expert_cls in (MoEExpertsCUTLASS, MoEExpertsCUTLASSFP8):
            # CUTLASS: setup_cutlass_gemm_meta is captured inside the graph
            # and reads batch_sizes from the static buffer directly.
            # We only need simple D2D copies — no m_indices, no fused kernel.
            self.static_input_hiddens[:num_tokens].copy_(hidden_states)
            self.static_input_batch_sizes.copy_(batch_sizes)
        else:
            # DeepGEMM: fused kernel copies hiddens + batch_sizes + m_indices
            fused_copy_and_pad_cuda(
                hidden_states, batch_sizes, m_indices,
                self.static_input_hiddens, self.static_input_batch_sizes, self.static_input_m_indices,
                bucket_size,
            )
    
    def _get_bucket_by_num_tokens(self, batch_size: int):
        for size in self.graph_batch_sizes:
            if size >= batch_size:
                return size
        assert False, f"No available graph for batch size={batch_size}"
    
    def capture(self):
        start_time = time.perf_counter()
        layer_time_elapse = []
        layer_memory_elapse = []
        get_logger().info(f"Captureing CUDA graphs for experts, bsz {self.graph_batch_sizes}")

        for graph_batch_size in self.graph_batch_sizes:
            graph_list = self.graphs[graph_batch_size]
            
            bsz_start_time = time.perf_counter()

            hiddens, batch_sizes, m_indices = make_expert_dummy_inputs(
                batch_size=graph_batch_size,
                hidden_size=self.model_config.hidden_size,
                num_experts_per_rank=self.local_num_experts,
                expert_ids=self.expert_ids,
            )

            self.cuda_graph_preprocess(hiddens, batch_sizes, m_indices, graph_batch_size)

            bsz_start_time = time.perf_counter()
            free_memory_before, _ = torch.cuda.mem_get_info()

            for layer_id in self.model_config.layer_ids:
                graph = graph_list[layer_id]

                batch = ExpertForwardBatch(
                            layer_id=layer_id,
                            data=self.static_input_hiddens[:graph_batch_size],
                            num_tokens=graph_batch_size,
                            meta_c=None,
                            proc_func=None,
                            post_proc_func=None,
                            batch_sizes=self.static_input_batch_sizes,
                            m_indices=self.static_input_m_indices[:graph_batch_size]
                        )
                
                for _ in range(2):
                    _ = self.experts_executor.execute_eager(batch)
                
                time_before_capture = time.perf_counter()
                
                if layer_id == 0:
                    with torch.cuda.graph(graph):
                        outputs = self.experts_executor.execute_eager(batch)
                else:
                    with torch.cuda.graph(graph, pool=graph_list[0].pool()):
                        outputs = self.experts_executor.execute_eager(batch)
                
                self.static_outputs[graph_batch_size].append(outputs)
                time_after_capture = time.perf_counter()
                if layer_id == 0 and graph_batch_size > 1:
                    get_logger().info(f"Time taken to capture graph: {time_after_capture - time_before_capture} seconds.")
                
            bsz_end_time = time.perf_counter()
            layer_time_elapse.append(bsz_end_time - bsz_start_time)
            free_memory_after, _ = torch.cuda.mem_get_info()
            layer_memory_elapse.append(round((free_memory_before - free_memory_after) / (1024 ** 3), 2))
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        get_logger().info(f"Expert cuda graphs captured in {end_time - start_time} seconds.")
        
    def test_graph(self):
        for layer_id in self.model_config.layer_ids:
            for bs in self.graph_batch_sizes:
                hiddens, batch_sizes, m_indices = make_expert_dummy_inputs(
                    bs,
                    self.model_config.hidden_size,
                    self.local_num_experts,
                    self.expert_ids,
                )
                
                self.run(layer_id, hiddens, batch_sizes, m_indices)

    def run(self, layer_id: int, hiddens: Tensor, batch_sizes: Tensor, m_indices: Tensor) -> Tensor:

        num_tokens = hiddens.shape[0]
        batch_size_bucket = self._get_bucket_by_num_tokens(num_tokens)

        try:
            self.cuda_graph_preprocess(hiddens, batch_sizes, m_indices, batch_size_bucket)
            self.graphs[batch_size_bucket][layer_id].replay()

            outputs = self.static_outputs[batch_size_bucket][layer_id]
        except Exception as e:
            get_logger().error(f"Error in expert run: {e}, layer_id {layer_id}, graph_bsz {batch_size_bucket}")
            raise e
        
        return outputs[ : num_tokens]
        
