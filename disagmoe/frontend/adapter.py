import torch

from disagmoe.frontend.datatypes import TensorBatch, AttentionBatchMetadata, SloStat

from typing import Tuple, List, Dict, Optional

class Scheduler:

    def wait_for_new_requests(self) -> None:
        ...

    def schedule(self, stream: Optional[torch.cuda.Stream] = None) -> TensorBatch:
        ...
        
    def get_channel(self) -> "NcclGroupChannel":
        ...
        
    def set_max_batch_size(self, max_batch_size: int) -> None:
        ...
        
    def get_pool_snapshot(self) -> List[int]:
        ...

    def get_cur_queueing_delay(self) -> List[float]:
        ...
        
    def set_schedule_policy(self, policy: str) -> None:
        ...
        
    def set_schedule_block(self, step: int) -> None:
        ...

class MuDispatcher:
        
    def put(self, batch: TensorBatch, rank: int):
        ...

class Tokenizer:
    
    def put_request(self, req_id: int, init_prefill_len: int, tensor: torch.Tensor, dp_rank: int) -> None:
        ...
        
    def start(self):
        ...
        
class Sampler:
    
    def start(self):
        ...
        
    def wait_slo_stats(self, n_request: int) -> Dict[int, SloStat]:
        ...
    
    def fetch_finished_slo_stats(self) -> List[SloStat]:
        ...
        
    def reset(self) -> None:
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
        
    def update_block_table(self, meta_c: AttentionBatchMetadata, decode_seq_lens: List[int]) -> None:
        ...
        
    def prepare_block_table(self, meta_c: AttentionBatchMetadata, decode_seq_lens: List[int]) -> torch.Tensor:
        ...
        
class NcclGroupChannel:
    
    def all_reduce(self, tensor_buf: int, shape: List[int]) -> None:
        ...
        
    def all_gather(self, tensor_buf: int, shape: List[int], dim: int = -1) -> None:
        ...