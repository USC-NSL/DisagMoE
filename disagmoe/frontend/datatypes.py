from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ChannelInfo:
    expert_ids: List[int]
    attn_layer_ids: List[int]
    
    def is_sampler_channel(self) -> bool:
        ...

@dataclass
class TokenMetadata:
    req_id: int
    exp_id: int
    first_attn_id: int
    prefill_pos: int

@dataclass
class Metadata:
    shape: List[int]
    dtype: str
    layer_id: int
    infos: List[TokenMetadata]
    prompt_lens: Dict[int, int]

    def step_layer(self) -> None:
        ...

    def update_exp_ids(self, 
                       new_exp_ids: List[int], 
                       required_sort: bool = True) -> None:
        ...

@dataclass
class TensorBatch:
    data: int       # pointer
    metadata: int # pointer
