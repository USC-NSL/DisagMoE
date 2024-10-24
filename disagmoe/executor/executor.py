import torch
from torch import Tensor

from typing import override, Tuple
from time import sleep
from enum import Enum

from vllm.attention import AttentionMetadata

from disagmoe.models.attention import MoEAttention
from disagmoe.models.experts import MoEExperts
from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.utils.utils import nvtx_range


class ExecutorType(Enum):
    ATTENTION_EXEC = 1
    EXPERTS_EXEC = 2
    
class Executor:
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.num_layers = model_config.num_layers
        self.layer_mappings = {
            id: i for i, id in enumerate(model_config.layer_ids)
        }
    
    def execute(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

class AttnExecutor(Executor):

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig):
        super().__init__(model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.operators = [
            MoEAttention(
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
            ) for _ in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
        assert self.cache_config.num_gpu_blocks > 0, "Should specify num gpu blocks for cache config"
        
        self._make_kv_cache(
            self.num_layers,
            self.cache_config.num_gpu_blocks,
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
                attn_metadata: AttentionMetadata) -> Tuple[Tensor, Tensor]:
        vid = self.layer_mappings[layer_id]
        operator = self.operators[vid]
        outputs, topk_experts = operator.forward(
            positions, 
            hidden_states, 
            self.cache[vid], 
            attn_metadata
        )
        return outputs, topk_experts

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