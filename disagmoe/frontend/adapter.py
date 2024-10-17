import torch

from disagmoe.frontend.datatypes import TensorBatch, AttentionBatchMetadata

from typing import Tuple, List

class Scheduler:

    def wait_for_new_requests(self) -> None:
        ...

    def schedule(self) -> TensorBatch:
        ...
        
    def prepare_block_table(meta: AttentionBatchMetadata, block_mgr: "BlockManager") -> List[List[int]]:
        ...

class MuDispatcher:
        
    def put(self, tensor: int, meta):
        ...

class Tokenizer:
    
    def put_request(self, tensor_buf: int, shape: Tuple[int]):
        ...
        
    def start():
        ...
        
class Sampler:
    
    def start():
        ...
        
class BlockManager:
    
    # def can_allocate(seq_len: int) -> bool:
    #     ...
        
    # def allocate(seq_id: int, seq_len: int) -> List[int]:
    #     ...
        
    def free(seq_id: int) -> None:
        ...
        
    def can_append() -> bool:
        ...
        
    def append_block(seq_id: int) -> None:
        ...
        
    def num_free_blocks() -> int:
        ...
    
    def get_seq_block_list(seq_id: int) -> List[int]:
        ...
        
    def has_seq_block_list(seq_id: int) -> bool:
        ...
        
    def append_tokens(seq_id: int, context_len: int, num_tokens: int) -> None:
        ...
        
    def allocate(seq_id: int, seq_len: int) -> None:
        ...
        