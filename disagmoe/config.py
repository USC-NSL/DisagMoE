from math import exp
from dataclasses import dataclass
from typing import Optional, List

import vllm
import torch
import vllm.config

@dataclass
class ModelConfig:    
    hidden_size: int
    num_layers: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    num_experts: int
    intermediate_size: int
    dtype: torch.dtype
    ep_size: int = 1 # default to 1
    tp_size: int = 1
    dp_size: int = 1
    rank: int = 0
    layer_ids: Optional[List[int]] = None
    top_k: int = 1
    max_seq_len: int = 4096
    
    # Attention-specific quantization option for QKV projection
    # e.g., "fp8" or None
    attn_qkv_quant: Optional[str] = None
    # MoE experts linear quantization option (applies to MoEExpertsSerial only)
    # e.g., "fp8" or None
    moe_linear_quant: Optional[str] = None
    
    # Shared expert configuration
    num_shared_experts: int = 0
    shared_expert_intermediate_size: Optional[int] = None
    
    @property
    def num_experts_per_rank(self):
        return self.num_experts // self.ep_size
    
@dataclass
class EngineConfig:
    # Unified (colocate) scheduler configuration.
    unified_scheduler_type: str
    defrag_weight_decay: float
    defrag_lookahead_steps: int
    defrag_lookback_steps: int

    enable_cuda_graph_attn: bool = False
    enable_cuda_graph_expert: bool = False
    enable_grouped_gemm: bool = False
    less_than_sm90: bool = False
    
    max_batch_size_attn: int = 160
    max_batch_size_expert: int = 512
    max_attn_graph_bsz: int = 160
    max_pending_sends: int = 16
    
    # FIXME(hogura|20250110): temporary field, should be moved to other place
    enable_trace: bool = False

    enable_advanced_logging: bool = False
    advanced_logging_dir: str = "./advanced_logs"
    advanced_logging_sample_rate: float = 0.1
    
@dataclass
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
    ) -> None:
        super().__init__(block_size, gpu_memory_utilization, 
                         swap_space, cache_dtype, num_gpu_blocks_override, 
                         sliding_window, enable_prefix_caching, cpu_offload_gb)

mixtral_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 32,
    head_dim = 128,
    num_heads = 32,
    num_kv_heads = 8,
    num_experts = 8,
    intermediate_size = 14336,
    dtype = torch.bfloat16,
    ep_size = 8,
    top_k = 2,
)

duo_expert_mixtral = ModelConfig(
    hidden_size = 4096,
    num_layers = 32,
    head_dim = 128,
    num_heads = 32,
    num_kv_heads = 8,
    num_experts = 2,
    intermediate_size = 14336,
    dtype = torch.bfloat16,
    ep_size = 2,
)

qwen3_235b_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 94,
    head_dim = 128,
    num_heads = 64,
    num_kv_heads = 4,
    num_experts = 128,
    intermediate_size = 1536,
    dtype = torch.bfloat16,
    top_k = 8,
)

qwen3_30b_config = ModelConfig(
    hidden_size = 2048,
    num_layers = 48,
    head_dim = 128,
    num_heads = 32,
    num_kv_heads = 4,
    num_experts = 128,
    intermediate_size = 768,
    dtype = torch.bfloat16,
    top_k = 8,
)

gptoss_120b_config = ModelConfig(
    hidden_size = 2880,
    num_layers = 36,
    head_dim = 64,
    num_heads = 64, # original 36
    num_kv_heads = 8, # original 8
    num_experts = 128,
    intermediate_size = 2880,
    dtype = torch.bfloat16,
    top_k = 4,
)

glm45air_106b_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 45,
    head_dim = 128,
    num_heads = 96,
    num_kv_heads = 8,
    num_experts = 128,
    intermediate_size = 1408,
    dtype = torch.bfloat16,
    top_k = 8,
)

glm45air_half_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 23,
    head_dim = 128,
    num_heads = 96,
    num_kv_heads = 8,
    num_experts = 128,
    intermediate_size = 1408,
    dtype = torch.bfloat16,
    top_k = 8,
)