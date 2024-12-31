from dataclasses import dataclass
from typing import List, Dict, Tuple
from disagmoe.utils.constants import CPS
import torch

@dataclass
class ChannelInfo:
    expert_ids: List[Tuple[int, int]]
    attn_layer_ids: List[int]
    attn_dp_rank: int
    
    def is_sampler_channel(self) -> bool:
        ...
        
    def to_c(self) -> "ChannelInfo_C":
        from disagmoe_c import ChannelInfo as ChannelInfo_C
        return ChannelInfo_C(
            self.expert_ids,
            self.attn_layer_ids,
            self.attn_dp_rank
        )

@dataclass
class TokenMetadata:
    req_id: int
    exp_id: int
    attn_dp_rank: int
    prefill_pos: int

@dataclass
class Metadata:
    shape: List[int]
    dtype: str
    layer_id: int
    # infos: List[TokenMetadata]
    req_ids: List[int]
    exp_ids: List[int]
    attn_dp_ranks: List[int]
    prefill_poss: List[int]

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
        
    @staticmethod
    def from_c(meta_c: "Metadata_C") -> "Metadata":
        return Metadata(
            meta_c.shape,
            meta_c.dtype,
            meta_c.layer_id,
            meta_c.req_ids,
            meta_c.exp_ids,
            meta_c.attn_dp_ranks,
            meta_c.prefill_poss
        )
        
    def to_c(self) -> "Metadata_C":
        from disagmoe_c import Metadata as Metadata_C
        meta_c = Metadata_C(self.shape)
        meta_c.dtype = self.dtype
        meta_c.layer_id = self.layer_id
        meta_c.req_ids = self.req_ids
        meta_c.exp_ids = self.exp_ids
        meta_c.attn_dp_ranks = self.attn_dp_ranks
        meta_c.prefill_poss = self.prefill_poss
        return meta_c
    

@dataclass
class TensorBatch:
    data: torch.Tensor
    metadata: Metadata
    
    @staticmethod
    def from_c(batch_c: "TensorBatch_C") -> "TensorBatch":
        return TensorBatch(
            batch_c.data,
            batch_c.metadata
        )

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
    attn_dp_ranks: List[int]
    
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
        attn_meta.attn_dp_ranks = self.attn_dp_ranks
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
            meta_c.expert_ids,
            meta_c.attn_dp_ranks
        )
        
@dataclass
class SloStat:
    req_id: int
    t_prefill: int
    t_decode: int
    t_tokens: List[int]
    
    @staticmethod
    def from_c(stat_c: "SloStat_C") -> "SloStat":
        return SloStat(
            stat_c.req_id,
            stat_c.t_prefill / CPS, # NOTE: consider how to deal with prefill time
            (stat_c.t_decode - stat_c.t_prefill) / CPS,
            [(x - y) / CPS for x, y in zip(stat_c.t_tokens[1:], stat_c.t_tokens[:-1])]
        )
        
@dataclass
class TraceContext:
    msg: str
    t_start: float
    t_dur: float
    track_id: int
    
    @staticmethod
    def from_c(ctx_c: "TraceContext_C") -> "TraceContext":
        return TraceContext(
            ctx_c.msg,
            ctx_c.t_start,
            ctx_c.t_dur,
            ctx_c.track_id
        )