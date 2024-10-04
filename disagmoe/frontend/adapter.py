import torch

from disagmoe.frontend.datatypes import TensorBatch

class Scheduler:

    def wait_for_new_requests(self) -> None:
        ...

    def schedule(self) -> TensorBatch:
        ...

class MuDispatcher:
        
    def put(self, tensor: torch.Tensor, meta):
        ...
