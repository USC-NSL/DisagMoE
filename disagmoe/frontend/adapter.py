import torch

from disagmoe.frontend.datatypes import TokenBatch, BatchMetadata, SloStat

from typing import Tuple, List, Dict, Optional

class Scheduler:

    class ScheduleTrace:
        batch: TokenBatch
        pool_snapshot: List[int]

    def schedule(self) -> TokenBatch:
        ...

    def schedule_trace(self) -> ScheduleTrace:
        ...
        
    def get_pool_snapshot(self) -> List[int]:
        ...
        
    def get_topk_pool_snapshot(self) -> List[int]:
        ...

    def set_schedule_policy(self, policy: str) -> None:
        ...
        
    def set_schedule_block(self, step: int) -> None:
        ...
        
    def set_schedule_token_threshold(self, attn_token_threshold: int, expert_token_threshold: int) -> None:
        ...

class MuDispatcher:
        
    def put(self, batch: TokenBatch, rank: int):
        ...

class MuPool:
    
    def put_batch(self, batch: TokenBatch) -> None:
        ...
        
class BlockManager:
    
    # def can_allocate(seq_len: int) -> bool:
    #     ...
        
    # def allocate(seq_id: int, seq_len: int) -> List[int]:
    #     ...
        
    def can_append(self) -> bool:
        ...
        
    def append_block(self, seq_id: int) -> None:
        ...
        
    def num_free_blocks(self) -> int:
        ...
    
    def get_seq_block_list(self, seq_id: int) -> List[int]:
        ...
        
    def has_seq_block_list(self, seq_id: int) -> bool:
        ...
        
    def append_tokens(self, seq_id: int, context_len: int, num_tokens: int) -> None:
        ...
        
    def allocate(self, seq_id: int, seq_len: int) -> None:
        ...
        
    def update_block_table(self, meta_c: BatchMetadata, decode_seq_lens: List[int]) -> None:
        ...
        
    def prepare_block_table(self, meta_c: BatchMetadata, decode_seq_lens: List[int]) -> torch.Tensor:
        ...
