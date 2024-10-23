from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ChannelInfo:
    expert_ids: List[Tuple[int, int]]
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
        
    def get_expert_batch_sizes(self, n_epxert: int) -> List[int]:
        ...

@dataclass
class TensorBatch:
    data: int       # pointer
    metadata: int # pointer

@dataclass
class AttentionBatchMetadata:
    layer_id: int
    shape: List[int]
    dtype: str
    
    num_prefill_seqs: int
    num_prefill_tokens: int
    num_decode_tokens: int
    seq_ids: List[int]
    
    prefill_seq_len: List[int]
    prefill_query_len: List[int]
    
    expert_ids: List[int]   # NOTE(hogura|20241014): internally uint8
    
    def to_metadata() -> Metadata:
        ...
        
@dataclass
class SloStat:
    t_prefill: int
    t_decode: int
    t_tokens: List[int]