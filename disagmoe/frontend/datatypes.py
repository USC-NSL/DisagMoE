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
    init_prefill_len: int

@dataclass
class Metadata:
    shape: List[int]
    dtype: str
    layer_id: int
    req_ids: List[int]
    exp_ids: List[int]
    attn_dp_ranks: List[int]
    init_prefill_lens: List[int]
    topk_weights: List[float]
    
    def get_dp_rank(self) -> int:
        ...
        
    def get_expert_id(self) -> int:
        ...

    def num_tokens(self) -> int:
        ...

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

    def duplicate_topk(self, topk) -> None:
        ...
    
    def shrink_topk(self, topk: int) -> None:
        ...
    
    def select_indices(self, indices: List[int]) -> "Metadata":
        ...
        
    def set_finish_signal(self, continue_ids: List[int]) -> None:
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
            meta_c.init_prefill_lens
        )
        
    def to_c(self) -> "Metadata_C":
        from disagmoe_c import Metadata as Metadata_C
        meta_c = Metadata_C(self.shape)
        meta_c.dtype = self.dtype
        meta_c.layer_id = self.layer_id
        meta_c.req_ids = self.req_ids
        meta_c.exp_ids = self.exp_ids
        meta_c.attn_dp_ranks = self.attn_dp_ranks
        meta_c.init_prefill_lens = self.init_prefill_lens
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
    
    init_prefill_lens: List[int]
    
    expert_ids: List[int]   # NOTE(hogura|20241014): internally uint8

    topk_weights: List[float]
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
        attn_meta.init_prefill_lens = self.init_prefill_lens
        attn_meta.expert_ids = self.expert_ids
        attn_meta.topk_weights = self.topk_weights
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
            meta_c.init_prefill_lens,
            meta_c.expert_ids,
            meta_c.topk_weights,
            meta_c.attn_dp_ranks
        )

    def shrink_topk(self, topk: int) -> None:
        ...
        
@dataclass
class SloStat:
    req_id: int
    
    # all time in seconds
    t_prefill: float
    t_prefill_std: float
    t_decode: float
    t_tokens: List[float]
    
    @staticmethod
    def from_c(stat_c: "SloStat_C") -> "SloStat":
        ms_to_s = 1e-3
        return SloStat(
            stat_c.req_id,
            stat_c.t_prefill * ms_to_s,
            stat_c.t_prefill_std * ms_to_s,
            (stat_c.t_decode - stat_c.t_prefill) * ms_to_s,
            [(x - y) * ms_to_s for x, y in zip(stat_c.t_tokens[1:], stat_c.t_tokens[:-1])]
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
        
@dataclass
class SamplerStepInfo:
    num_tokens: int
    time_stamp: int
    
    @staticmethod
    def from_c(step_c: "SamplerStepInfo_C") -> "SamplerStepInfo":
        return SamplerStepInfo(
            step_c.num_tokens,
            step_c.time_stamp
        )