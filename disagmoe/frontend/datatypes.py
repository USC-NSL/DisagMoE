from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Union
import torch
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from disagmoe.utils.gdr_context import GdrContext
from disagmoe_c import (
    BatchMetadata as BatchMetadata_C,
    ChannelInfo as ChannelInfo_C,
    TokenBatch as TokenBatch_C,
    TraceContext as TraceContext_C,
)

@dataclass
class ChannelInfo:
    expert_ids: List[Tuple[int, int]]
    attn_layer_ids: List[int]
    attn_dp_rank: int
        
    def to_c(self) -> "ChannelInfo_C":
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
    topk_weight: float

@dataclass
class BatchMetadata:
    shape: List[int]
    dtype: str
    layer_id: int
    req_ids: List[int]
    exp_ids: List[int]
    topk_weights: List[float]
    attn_dp_ranks: List[int]
    init_prefill_lens: List[int]
    
    # Only used in attention batch - optional fields
    num_prefill_seqs: Optional[int] = None
    num_prefill_tokens: Optional[int] = None
    num_decode_tokens: Optional[int] = None
    
    def is_attention(self) -> bool:
        ...
    
    def is_expert(self) -> bool:
        ...
    
    def is_tokenizer(self) -> bool:
        ...
    
    def num_tokens(self) -> int:
        ...
    
    def token_hidden_dim(self) -> int:
        ...
    
    def step_layer(self) -> None:
        ...
    
    def set_finish_signal(self, continue_ids: List[int]) -> None:
        ...
    
    def get_expert_batch_sizes(self, n_expert: int) -> List[int]:
        ...
    
    def get_expert_batch_sizes_cuda(self, n_expert: int, inner_exp_rank: List[int], tensor_cuda: torch.Tensor, stream_ptr: int) -> None:
        ...
        
    def get_finished_indices(self) -> List[int]:
        ...
    
    def permute_token_infos(self, positions: List[int]) -> None:
        ...
    
    def duplicate_topk(self, topk: int) -> None:
        ...
    
    def sort_by_attention(self) -> List[int]:
        ...
    
    def sort_by_expert(self) -> List[int]:
        ...
        
    def index_select(self, indices: List[int]) -> "BatchMetadata_C":
        ...
    
    @staticmethod
    def from_c(meta_c: "BatchMetadata_C") -> "BatchMetadata":
        return BatchMetadata(
            shape=meta_c.shape,
            dtype=meta_c.dtype,
            layer_id=meta_c.layer_id,
            req_ids=meta_c.req_ids,
            exp_ids=meta_c.exp_ids,
            topk_weights=meta_c.topk_weights,
            attn_dp_ranks=meta_c.attn_dp_ranks,
            init_prefill_lens=meta_c.init_prefill_lens,
            num_prefill_seqs=meta_c.num_prefill_seqs,
            num_prefill_tokens=meta_c.num_prefill_tokens,
            num_decode_tokens=meta_c.num_decode_tokens
        )
        
    def to_c(self) -> "BatchMetadata_C":
        meta_c = BatchMetadata_C()
        meta_c.shape = self.shape
        meta_c.dtype = self.dtype
        meta_c.layer_id = self.layer_id
        meta_c.req_ids = self.req_ids
        meta_c.exp_ids = self.exp_ids
        meta_c.topk_weights = self.topk_weights
        meta_c.attn_dp_ranks = self.attn_dp_ranks
        meta_c.init_prefill_lens = self.init_prefill_lens
        meta_c.num_prefill_seqs = self.num_prefill_seqs
        meta_c.num_prefill_tokens = self.num_prefill_tokens
        meta_c.num_decode_tokens = self.num_decode_tokens
        return meta_c
    
@dataclass
class TokenBatch:
    data: torch.Tensor
    metadata: BatchMetadata
    
    @staticmethod
    def from_c(batch_c: "TokenBatch_C") -> "TokenBatch":
        return TokenBatch(
            batch_c.data,
            batch_c.metadata
        )
        
    @staticmethod
    def make_c(data: torch.Tensor, metadata: BatchMetadata_C) -> "TokenBatch_C":
        batch = TokenBatch_C()
        batch.data = data
        batch.metadata = metadata
        return batch
    
@dataclass
class TokenBatchCWrapper:
    data: torch.Tensor
    metadata: BatchMetadata_C
    
    @staticmethod
    def from_c(batch_c: "TokenBatch_C") -> "TokenBatchCWrapper":
        return TokenBatchCWrapper(
            data=batch_c.data,
            metadata=batch_c.metadata
        )
    
    def to_c(self) -> "TokenBatch_C":
        batch = TokenBatch_C()
        batch.data = self.data
        batch.metadata = self.metadata
        return batch
        
@dataclass
class AttentionScheduleBatch:
    
    shape: List[int]
    dtype: str
    layer_id: int
    req_ids: List[int]
    init_prefill_lens: List[int]
    
    num_prefill_seqs: int
    num_prefill_tokens: int
    num_decode_tokens: int
    
    data: torch.Tensor
    meta_c: Optional[BatchMetadata_C] = None
    
    # used in engine and executor
    req_indices: Optional[List[int]] = None
    req_indices_tensor: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    seq_lens_tensor: Optional[torch.Tensor] = None
    
    def num_tokens(self) -> int:
        return self.num_decode_tokens + self.num_prefill_tokens
    
    @staticmethod
    def build(meta: BatchMetadata, data: torch.Tensor) -> "AttentionScheduleBatch":
        return AttentionScheduleBatch(
            shape=meta.shape,
            dtype=meta.dtype,
            layer_id=meta.layer_id,
            req_ids=meta.req_ids,
            init_prefill_lens=meta.init_prefill_lens,
            num_prefill_seqs=meta.num_prefill_seqs,
            num_prefill_tokens=meta.num_prefill_tokens,
            num_decode_tokens=meta.num_decode_tokens,
            data=data,
            meta_c=meta if isinstance(meta, BatchMetadata_C) else None
        )
        
    def to_metadata(self) -> BatchMetadata:
        return BatchMetadata(
            shape=self.shape,
            dtype=self.dtype,
            layer_id=self.layer_id,
            req_ids=self.req_ids,
            exp_ids=[],
            topk_weights=[],
            attn_dp_ranks=[],
            init_prefill_lens=self.init_prefill_lens,
            num_prefill_seqs=self.num_prefill_seqs,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=self.num_decode_tokens
        )
        
    def to_metadata_c(self) -> "BatchMetadata_C":
        return self.to_metadata().to_c()

@dataclass
class ForwardBatch:
    layer_id: int
    data: torch.Tensor
    num_tokens: int
    meta_c: BatchMetadata_C
    
    proc_func: Optional[Callable[["ForwardBatch"], "ForwardResult"]]
    post_proc_func: Optional[Callable[["ForwardBatch", "ForwardResult"], TokenBatchCWrapper]]
    
    def to_string(self) -> str:
        return f"ForwardBatch(layer_id={self.layer_id}, num_tokens={self.num_tokens}, {self.data.shape})"

@dataclass
class AttentionForwardBatch(ForwardBatch):
    positions: torch.Tensor
    metadata: FlashAttentionMetadata
    req_ids: Optional[List[int]] = None
    output_buffer: Optional[torch.Tensor] = None
    expert_ids_buffer: Optional[torch.Tensor] = None
    expert_weights_buffer: Optional[torch.Tensor] = None
    expert_ids_buffer_gdr: Optional[GdrContext] = None
    expert_weights_buffer_gdr: Optional[GdrContext] = None
    
    def to_string(self) -> str:
        return f"AttentionForwardBatch(layer_id={self.layer_id}, num_tokens={self.num_tokens}, {self.data.shape}, {self.req_ids})"

@dataclass
class ExpertForwardBatch(ForwardBatch):
    batch_sizes: Optional[torch.Tensor]
    m_indices: Optional[torch.Tensor]
    
    def to_string(self) -> str:
        return f"ExpertForwardBatch(layer_id={self.layer_id}, num_tokens={self.num_tokens}, {self.data.shape})"

@dataclass
class ForwardResult:
    hiddens: torch.Tensor
    sync_event: Optional[torch.cuda.Event]

@dataclass
class AttentionForwardResult(ForwardResult):
    expert_weights: Optional[torch.Tensor]
    expert_ids: Optional[torch.Tensor]

@dataclass
class ExpertForwardResult(ForwardResult):
    pass
    
@dataclass
class SloStat:
    req_id: int
    
    # all time in seconds
    t_prefill: float
    t_prefill_std: float
    t_decode: float
    t_tokens: List[float]
    # absolute token timestamps in seconds (preserved for time-window filtering)
    t_token_timestamps: List[float] = None
        
    def post_process(self) -> None:
        ms_to_s = 1e-3
        self.t_decode = (self.t_decode - self.t_prefill) * ms_to_s
        self.t_prefill = self.t_prefill * ms_to_s
        self.t_prefill_std = self.t_prefill_std * ms_to_s
        self.t_token_timestamps = [t * ms_to_s for t in self.t_tokens]
        self.t_tokens = [(x - y) * ms_to_s for x, y in zip(self.t_tokens[1:], self.t_tokens[:-1])]
        
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

@dataclass
class TokenizedRequest:
    
    req_id: int
    init_prefill_len: int
    max_output_len: int
    token_ids: List[int]

@dataclass
class BatchDecodeResult:
    
    req_ids: List[int]
    token_ids: List[int]
    is_eos: List[bool]