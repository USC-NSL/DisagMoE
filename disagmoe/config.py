from dataclasses import dataclass
import torch
@dataclass
class ModelConfig:
    
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    num_experts: int
    intermediate_size: int
    dtype: torch.dtype