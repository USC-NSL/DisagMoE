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
    
    def execute(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def forward(self, x: Tensor) -> Tensor:
        # NOTE(hogura|20241014): a dummy forward
        return x + 1

class AttnExecutor(Executor):

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig):
        super().__init__(model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.operator = MoEAttention(
            self.model_config.hidden_size, 
            self.model_config.num_heads, 
            self.model_config.num_kv_heads, 
            self.model_config.num_experts,
        )
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
        assert self.cache_config.num_gpu_blocks > 0, "Should specify num gpu blocks for cache config"
        
        self._make_kv_cache(
            self.cache_config.num_gpu_blocks,
            self.cache_config.block_size, 
            self.model_config.num_kv_heads, 
            self.model_config.hidden_size // self.model_config.num_heads,
        )
    
    def _make_kv_cache(self, num_blocks, block_size, num_heads, head_size):
        data_type = self.model_config.dtype
        self.cache = torch.zeros((2, num_blocks, block_size, num_heads, head_size), dtype=data_type)

    @override
    @nvtx_range("AttnExecutor.execute")
    def execute(self,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata) -> Tuple[Tensor, Tensor]:
        outputs, topk_experts = self.operator.forward(
            positions, 
            hidden_states, 
            self.cache, 
            attn_metadata
        )
        return outputs, topk_experts

class ExpertsExecutor(Executor):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.type = ExecutorType.EXPERTS_EXEC
        self.opertor = MoEExperts(
            self.model_config.hidden_size, 
            self.model_config.intermediate_size,
            self.model_config.num_experts_per_rank,
        )

    @override
    @nvtx_range("ExpertsExecutor.execute")
    def execute(self, hidden_states: Tensor, batch_sizes: Tensor) -> Tensor:
        outputs = self.opertor.forward(hidden_states, batch_sizes)
        return outputs