import torch
import torch.distributed as dist

from torch import Tensor

from typing import override, Tuple
from time import sleep
from enum import Enum

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig as VllmCacheConfig

from disagmoe.models.attention import MoEAttention
from disagmoe.models.experts import MoEExperts
from disagmoe.config import ModelConfig, CacheConfig as DmoeCacheConfig
from disagmoe.utils.utils import nvtx_range


class ExecutorType(Enum):
    ATTENTION_EXEC = 1
    EXPERTS_EXEC = 2
    
class Executor:
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.num_layers = len(model_config.layer_ids)
        self.layer_mappings = {
            id: i for i, id in enumerate(model_config.layer_ids)
        }
    
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
            gpu_memory_utilization=0, # useless in our case
            swap_space=0, #useless in our case
        )
        self.operators = [
            MoEAttention(
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                self.model_config.top_k,
                cache_config=self.vllm_cache_config,
            ) for _ in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
        assert self.cache_config.num_gpu_blocks > 0, "Should specify num gpu blocks for cache config"
        
        self._make_kv_cache(
            self.num_layers,
            self.cache_config.num_gpu_blocks + self.cache_config.num_reserved_blocks,
            self.cache_config.block_size, 
            self.model_config.num_kv_heads, 
            self.model_config.hidden_size // self.model_config.num_heads,
        )
    
    def _make_kv_cache(self, num_layers, num_blocks, block_size, num_heads, head_size):
        data_type = self.model_config.dtype
        self.cache = torch.zeros((num_layers, 2, num_blocks, block_size, num_heads, head_size), dtype=data_type)

    @override
    @nvtx_range("AttnExecutor.execute")
    def execute(self,
                layer_id: int,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata) -> Tuple[Tensor, Tensor, Tensor]:
        vid = self.layer_mappings[layer_id]
        operator = self.operators[vid]
        outputs, top_k_weights, topk_experts = operator.forward(
            positions, 
            hidden_states, 
            self.cache[vid], 
            attn_metadata
        )
        return outputs, top_k_weights, topk_experts
    
    @staticmethod
    def build(model_config: ModelConfig, cache_config: DmoeCacheConfig) -> "Executor":
        if model_config.tp_size > 1:
            return ParallelAttnExecutor(model_config, cache_config)
        else:
            return AttnExecutor(model_config, cache_config)

class ExpertsExecutor(Executor):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.type = ExecutorType.EXPERTS_EXEC
        self.operators = [
            MoEExperts(
                self.model_config.hidden_size, 
                self.model_config.intermediate_size,
                self.model_config.num_experts_per_rank,
            ) for _ in range(self.num_layers)
        ]

    @override
    @nvtx_range("ExpertsExecutor.execute")
    def execute(self, layer_id: int, hidden_states: Tensor, batch_sizes: Tensor) -> Tensor:
        vid = self.layer_mappings[layer_id]
        operator = self.operators[vid]
        outputs = operator.forward(hidden_states, batch_sizes)
        return outputs
    
class ParallelAttnExecutor(AttnExecutor):
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig):
        Executor.__init__(self, model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.operators = [
            MoEAttention(
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                tp_size=model_config.tp_size,
                tp_rank=model_config.rank,
            ) for _ in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
        assert self.cache_config.num_gpu_blocks > 0, "Should specify num gpu blocks for cache config"
        
        self._make_kv_cache(
            self.num_layers,
            self.cache_config.num_gpu_blocks + self.cache_config.num_reserved_blocks,
            self.cache_config.block_size, 
            self.model_config.num_kv_heads // self.model_config.tp_size, 
            self.model_config.hidden_size // self.model_config.num_heads,
        )