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
    # infos: List[TokenMetadata]
    req_ids: List[int]
    exp_ids: List[int]
    prefill_poss: List[int]
    prompt_lens: Dict[int, int]

    def step_layer(self) -> None:
        ...

    def update_exp_ids(self, 
                       new_exp_ids: List[int], 
                       exp_mappings: List[int]) -> None:
        ...
        
    def get_expert_batch_sizes(self, n_epxert: int) -> List[int]:
        ...
    
    def permute_token_infos(exp_mappings: List[int]) -> None:
        ...

    def sort_by_prefill_order(self) -> List[int]:
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
    
    def to_metadata(self) -> Metadata:
        ...
        
    def to_c(self) -> "AttentionBatchMetadata_C":
        from disagmoe_c import AttentionBatchMetadata as AttentionBatchMetadata_C
        attn_meta = AttentionBatchMetadata_C()
        attn_meta.layer_id = self.layer_id
        attn_meta.shape = self.shape
        attn_meta.dtype = self.dtype
        attn_meta.num_prefill_seqs = self.num_prefill_seqs
        attn_meta.num_prefill_tokens = self.num_prefill_tokens
        attn_meta.num_decode_tokens = self.num_decode_tokens
        attn_meta.seq_ids = self.seq_ids
        attn_meta.prefill_seq_len = self.prefill_seq_len
        attn_meta.prefill_query_len = self.prefill_query_len
        attn_meta.expert_ids = self.expert_ids
        return attn_meta
    
    @staticmethod
    def from_c(meta_c: "AttentionBatchMetadata_C") -> "AttentionBatchMetadata":
        return AttentionBatchMetadata(
            meta_c.layer_id,
            meta_c.shape,
            meta_c.dtype,
            meta_c.num_prefill_seqs,
            meta_c.num_prefill_tokens,
            meta_c.num_decode_tokens,
            meta_c.seq_ids,
            meta_c.prefill_seq_len,
            meta_c.prefill_query_len,
            meta_c.expert_ids
        )
        
@dataclass
class SloStat:
    t_prefill: int
    t_decode: int
    t_tokens: List[int]