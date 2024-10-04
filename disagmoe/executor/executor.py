import torch

from torch import Tensor
from time import sleep
from typing import override

class Executor:
    
    def __init__(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

class AttnExecutor(Executor):

    def __init__(self):
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        # TODO
        return x + 1

class FFNExecutor(Executor):

    def __init__(self):
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        # TODO
        return x + 1