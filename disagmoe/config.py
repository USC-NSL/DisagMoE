from dataclasses import dataclass
from typing import Optional, List

import vllm
import torch
import vllm.config

@dataclass
class ModelConfig:    
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    num_experts: int
    intermediate_size: int
    dtype: torch.dtype
    ep_size: int
    tp_size: int = 1
    rank: int = 0
    
    tp_enable_inter_group: bool = True
    enable_cuda_graph: bool = False
    
    @property
    def num_experts_per_rank(self):
        return self.num_experts // self.ep_size
    
    @property
    def layer_ids(self):
        return list(range(self.num_layers))
    
    
class CacheConfig(vllm.config.CacheConfig):
    
    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: float,
        cache_dtype: str,
        num_gpu_blocks_override: Optional[int] = None,
        sliding_window: Optional[int] = None,
        enable_prefix_caching: bool = False,
        cpu_offload_gb: float = 0,
        num_reserved_blocks: int = 0,
        num_gpu_blocks: Optional[int] = None,
    ) -> None:
        super().__init__(block_size, gpu_memory_utilization, 
                         swap_space, cache_dtype, num_gpu_blocks_override, 
                         sliding_window, enable_prefix_caching, cpu_offload_gb)
        self.num_reserved_blocks = num_reserved_blocks
        self.num_gpu_blocks = num_gpu_blocks
        

mixtral_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 32,
    num_heads = 32,
    num_kv_heads = 8,
    num_experts = 8,
    intermediate_size = 14336,
    dtype = torch.bfloat16,
    ep_size = 8,
)

duo_expert_mixtral = ModelConfig(
    hidden_size = 4096,
    num_layers = 3,
    num_heads = 32,
    num_kv_heads = 8,
    num_experts = 2,
    intermediate_size = 14336,
    dtype = torch.bfloat16,
    ep_size = 2,
)