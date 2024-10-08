import torch

from disagmoe.frontend.datatypes import TensorBatch

from typing import Tuple, List

class Scheduler:

    def wait_for_new_requests(self) -> None:
        ...

    def schedule(self) -> TensorBatch:
        ...

class MuDispatcher:
        
    def put(self, tensor: int, meta):
        ...

class Tokenizer:
    
    def put_request(self, tensor_buf: int, shape: Tuple[int]):
        ...