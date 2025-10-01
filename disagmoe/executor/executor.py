import torch
import torch.distributed as dist

from torch import Tensor
import numpy as np

from typing import Tuple, List, Union, Dict
from time import sleep
from enum import Enum

from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import CacheConfig as VllmCacheConfig

from disagmoe.env import ENV_VARS
from disagmoe.models.attention import MoEAttention
from disagmoe.models.experts import MoEExperts, MoEExpertsSerial
from disagmoe.config import ModelConfig, CacheConfig as DmoeCacheConfig
from disagmoe.utils.utils import nvtx_range, _log_memory_usage
from disagmoe.utils.logger import get_logger
from disagmoe.models.utils import make_dummy_meta, make_prefill_meta
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from disagmoe.block_manager.block_manager import GPUBlockManager, CPUBlockManager, BaseBlockManager
from disagmoe.block_manager.mem_pool import MHATokenToKVPool
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from disagmoe_c import prepare_batch_infos

class ExecutorType(Enum):
    ATTENTION_EXEC = 1
    EXPERTS_EXEC = 2
    
class Executor:
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.num_layers = len(model_config.layer_ids)
        self.layer_mappings = [0 for _ in range(max(model_config.layer_ids) + 1)]
        for i, id in enumerate(model_config.layer_ids):
            self.layer_mappings[id] = i
    
    def execute(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
class AttnExecutor(Executor):

    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig):
        super().__init__(model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.vllm_cache_config = VllmCacheConfig(
            cache_dtype="auto",
            block_size=cache_config.block_size,
            gpu_memory_utilization=0,
            swap_space=0,
        )
        self.enable_cuda_graph = self.model_config.enable_cuda_graph_attn
        self.cuda_graph_executor = None
        self.device = "cuda"
        self.block_mgr: BaseBlockManager = None
        
        self.init_model_and_cache()
        
    def init_model_and_cache(self, use_gpu_block_mgr: bool = False):
        _log_memory_usage("Setup device")
        free_memory, _ = torch.cuda.mem_get_info()
        self.init_gpu_memory = free_memory
        
        self.operators = [
            MoEAttention(
                layer_id,
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                self.model_config.top_k,
                cache_config=self.vllm_cache_config,
            ) for layer_id in range(self.num_layers)
        ]
        _log_memory_usage("After allocate parameters")
        
        assert not self.cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
        if self.cache_config.num_gpu_blocks is None:
            self.num_cache_blocks = self.determine_kv_cache_blocks()
            self.cache_config.num_gpu_blocks = self.num_cache_blocks
            get_logger().info(f"kv cache num_gpu_blocks: {self.cache_config.num_gpu_blocks}")
        else:
            self.num_cache_blocks = self.cache_config.num_gpu_blocks
            
        self.kv_cache = MHATokenToKVPool(
            self.num_cache_blocks,
            self.cache_config.block_size,
            self.model_config.dtype,
            self.model_config.num_kv_heads,
            self.model_config.hidden_size // self.model_config.num_heads,
            self.num_layers,
            self.device,
        )
        
        _log_memory_usage("After initialize cache")
        
        # TODO: fix this magic number
        self.max_running_reqs = self.num_cache_blocks * self.cache_config.block_size // 100 + 1
        
        if use_gpu_block_mgr:
            self.block_mgr = GPUBlockManager(self.model_config, self.cache_config, self.max_running_reqs, self.device)
        else:
            self.block_mgr = CPUBlockManager(self.model_config, self.cache_config, self.max_running_reqs, self.device)
        
    def get_num_cache_blocks(self):
        return self.num_cache_blocks
    
    def get_block_mgr(self):
        return self.block_mgr

    def memory_profile(self, batch_size: int):
        # use prefill to simulate a batch decoding to get memory profile
        attn_metadata = make_prefill_meta(batch_size, self.cache_config.block_size)
        kv_cache = torch.tensor([])
        for layer_id in range(self.num_layers):
            positions = torch.ones(batch_size, dtype=torch.long, device=self.device)
            hidden_states = torch.randn((batch_size, self.model_config.hidden_size), dtype=self.model_config.dtype)
            operator = self.operators[layer_id]
            operator.forward(positions, hidden_states, kv_cache, attn_metadata)
            
    def determine_kv_cache_blocks(self) -> int:
        torch.cuda.empty_cache()
                
        self.memory_profile(self.model_config.max_batch_size_attn)      
        torch.cuda.synchronize()
        
        _log_memory_usage("After profile run")
        
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = self.init_gpu_memory - free_gpu_memory
        cache_block_size = self.model_config.hidden_size // self.model_config.num_heads \
                            * self.model_config.num_kv_heads * self.cache_config.block_size * 2 * 2 # 2 for kv, 2 for fp16/bf16
        
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) 
            // cache_block_size // len(self.model_config.layer_ids)
        )
        
        return num_gpu_blocks
    
    def build_cuda_graph_executor(self):
        if self.enable_cuda_graph:
            self.cuda_graph_executor = CUDAGraphAttnExecutor(self.model_config, self.cache_config, self)
            self.cuda_graph_executor.create_cuda_graph_buffers()
            self.cuda_graph_executor.capture()
            _log_memory_usage("After build CUDA graphs")
            
    def warmup(self, batch_size: int):
        get_logger().info("Attention warmup start")
        input = torch.zeros((batch_size, self.model_config.hidden_size), device="cuda")
        positions = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        meta_py = make_dummy_meta(0, batch_size, 256)
        meta = self.block_mgr.pack_flash_attn_metadata(meta_py.to_c(), meta_py, dummy_cache=True)
        for layer_id in self.model_config.layer_ids:
            get_logger().info(f"Attention warmup layer {layer_id} start")
            for _ in range(2):
                self.execute_eager(layer_id, positions, input, meta)
            get_logger().info(f"Attention warmup layer {layer_id} done")
                
        get_logger().info("Attention warmup done")
    
    def execute_eager(self,
                layer_id: int,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                attn_metadata: FlashAttentionMetadata) -> Tuple[Tensor, Tensor, Tensor]:
        vid = self.layer_mappings[layer_id]
        outputs, topk_weights, topk_ids = self.operators[vid].forward(
            positions, 
            hidden_states, 
            self.kv_cache.get_kv_buffer(vid), 
            attn_metadata
        )
        return outputs, topk_weights, topk_ids
    
    @nvtx_range("AttnExecutor.execute")
    def execute(self, layer_id: int,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                attn_metadata: FlashAttentionMetadata) -> Tuple[Tensor, Tensor, Tensor]:
        if self.enable_cuda_graph and attn_metadata.use_cuda_graph and attn_metadata.num_decode_tokens <= self.attn_max_batch_size:
            return self.cuda_graph_executor.run(layer_id, positions, hidden_states, attn_metadata)
        else:
            return self.execute_eager(layer_id, positions, hidden_states, attn_metadata)
    
    @staticmethod
    def build(model_config: ModelConfig, cache_config: DmoeCacheConfig) -> "Executor":
        if model_config.tp_size > 1:
            return ParallelAttnExecutor(model_config, cache_config)
        else:
            return AttnExecutor(model_config, cache_config)
        
class CUDAGraphAttnExecutor:
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig, attn_executor: AttnExecutor):
        self.model_config = model_config
        self.cache_config = cache_config
        self.attn_executor = attn_executor
        
    def create_cuda_graph_buffers(self):
        assert self.model_config.enable_cuda_graph_attn
        batch_size = self.model_config.max_batch_size_attn
        self.graphs: Dict[int, List[torch.cuda.CUDAGraph]] = {}
        self.static_outputs: Dict[int, List[Tuple[Tensor]]] = {}

        self.static_input = torch.zeros((batch_size, self.model_config.hidden_size), device="cuda")
        self.static_positions = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        self.static_block_table = torch.zeros(
            (batch_size, self.model_config.max_seq_len // self.cache_config.block_size), 
            dtype=torch.int32, device="cuda")
        self.static_slot_mapping = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")

        self.static_batch_info = torch.zeros((batch_size + batch_size + (batch_size + 1)), dtype=torch.int32, device="cuda")
        self.static_seq_lens = self.static_batch_info[ : batch_size]
        self.static_context_lens = self.static_batch_info[batch_size : batch_size + batch_size]
        self.static_seq_start_loc = self.static_batch_info[batch_size + batch_size : ]

        self.static_batch_infos: Dict[int, Tensor] = {}

        self.graph_batch_sizes = list(range(max(self.model_config.graph_stride, self.model_config.ep_size),
                                            batch_size + 1,
                                            self.model_config.graph_stride))
        self.graph_batch_sizes = [1] + self.graph_batch_sizes

        for layer_id in self.model_config.layer_ids:
            self.graphs[layer_id] = [torch.cuda.CUDAGraph() for _ in self.graph_batch_sizes]
            self.static_outputs[layer_id] = []
        
        for bs in self.graph_batch_sizes:
            self.static_batch_infos[bs] = torch.zeros((bs + bs + (bs + 1)), dtype=torch.int32, device="cuda")

    def capture(self):
        for layer_id in self.model_config.layer_ids:
            for graph, graph_batch_size in zip(self.graphs[layer_id], self.graph_batch_sizes):
                meta_py = make_dummy_meta(0, graph_batch_size)
                attn_meta = self.attn_executor.block_mgr.pack_flash_attn_metadata(meta_py.to_c(), meta_py, dummy_cache=True)

                def run_once() -> Tuple[Tensor, Tensor, Tensor]:
                    return self.attn_executor.execute(
                        layer_id, self.static_positions[ : graph_batch_size], 
                        self.static_input[ : graph_batch_size], attn_meta
                    )

                for _ in range(2):
                    # warmup
                    torch.cuda.synchronize()
                    run_once()
                    torch.cuda.synchronize()

                with torch.cuda.graph(graph):
                    outputs = run_once()
                    
                torch.cuda.synchronize()

                self.static_outputs[layer_id].append(outputs)
                
                # warmup for the actual execution
                graph.replay()
                    
                torch.cuda.synchronize()

        print("cuda graph captured")

        self.test_graph()
        
        print("cuda graph tested")

    def test_graph(self):
        for layer_id in self.model_config.layer_ids:
            for bs in range(1, self.model_config.max_batch_size_attn + 1):
                meta_py = make_dummy_meta(0, bs)
                meta = self.attn_executor.block_mgr.pack_flash_attn_metadata(meta_py.to_c(), meta_py, dummy_cache=True)
                hiddens, expert_weights, expert_ids = self.run(layer_id, torch.zeros(bs, dtype=torch.long, device="cuda"), torch.randn(bs, self.model_config.hidden_size, device="cuda"), meta)
                torch.cuda.synchronize()
                _, reorder_ids = torch.sort(expert_ids.view(-1), stable=True)

    def _get_graph_by_batch_size(self, batch_size: int):
        for i, size in enumerate(self.graph_batch_sizes):
            if size >= batch_size:
                return i, size
        assert False, f"No available graph for batch size={batch_size}"

    def run(self, layer_id: int, positions: torch.Tensor, hidden_states: torch.Tensor, meta: FlashAttentionMetadata) -> Tuple[Tensor, Tensor, Tensor]:
        meta.use_cuda_graph = True
        
        num_tokens = hidden_states.shape[0]
        graph_id, batch_size = self._get_graph_by_batch_size(num_tokens)
        self.static_input[ : num_tokens].copy_(hidden_states)
        self.static_positions[ : num_tokens].copy_(positions)
        self.static_slot_mapping[ : num_tokens].copy_(meta.slot_mapping)

        max_num_blocks = meta.block_tables.shape[1]
        self.static_block_table[ : num_tokens, : max_num_blocks].copy_(meta.block_tables)

        # static_batch_info = self.static_batch_infos[batch_size]

        # seq_lens_cuda  = static_batch_info[ : batch_size]
        # context_lens_cuda = static_batch_info[batch_size : batch_size + batch_size]
        # seq_start_loc_cuda = static_batch_info[batch_size + batch_size : ]
        # seq_lens_cuda[ : num_tokens].copy_(meta.seq_lens_tensor)
        # context_lens_cuda[ : num_tokens].copy_(meta.context_lens_tensor)
        # seq_start_loc_cuda[ : num_tokens + 1].copy_(meta.seq_start_loc)
        self.static_seq_lens[ : num_tokens].copy_(meta.seq_lens_tensor)
        self.static_context_lens[ : num_tokens].copy_(meta.context_lens_tensor)
        self.static_seq_start_loc[ : num_tokens + 1].copy_(meta.seq_start_loc)

        self.graphs[layer_id][graph_id].replay()

        outputs, topk_weights, topk_ids = self.static_outputs[layer_id][graph_id]

        return outputs[ : num_tokens], topk_weights[ : num_tokens], topk_ids[ : num_tokens]
        
class ExpertsExecutor(Executor):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        expert_cls = MoEExperts if model_config.enable_grouped_gemm else MoEExpertsSerial
        self.type = ExecutorType.EXPERTS_EXEC
        self.operators = [
            expert_cls(
                self.model_config.hidden_size,
                self.model_config.intermediate_size,
                self.model_config.num_experts_per_rank,
                max_batch_size=self.model_config.max_batch_size_expert
            ) for _ in range(self.num_layers)
        ]
        
    
    def warmup(self, batch_size: int):
        self._static_bs_cuda = torch.zeros((self.model_config.num_experts_per_rank, ), dtype=torch.int64, device="cuda")
        
        input = torch.zeros((batch_size, self.model_config.hidden_size), device="cuda")
        batch_sizes = torch.tensor([batch_size // self.model_config.num_experts_per_rank] * self.model_config.num_experts_per_rank,
            dtype=torch.int64,
            # NOTE(hogura|20241014): cuBLAS grouped_gemm requires batch_sizes to be on cpu
            device="cuda" if ENV_VARS["GROUPED_GEMM_CUTLASS"] else "cpu")
        for layer_id in self.model_config.layer_ids:
            for _ in range(2):
                _ = self.execute(layer_id, batch_size, input, batch_sizes)

    @nvtx_range("ExpertsExecutor.execute")
    def execute(self, layer_id: int, num_tokens: int, hidden_states: Tensor, batch_sizes: Tensor) -> Tensor:
        vid = self.layer_mappings[layer_id]
        operator = self.operators[vid]
        outputs = operator.forward(num_tokens, hidden_states, batch_sizes)
        return outputs
    
class ParallelAttnExecutor(AttnExecutor):
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig):
        Executor.__init__(self, model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.operators = [
            MoEAttention(
                layer_id,
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                tp_size=model_config.tp_size,
                tp_rank=model_config.rank,
            ) for layer_id in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
