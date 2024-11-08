from disagmoe.config import ModelConfig
from disagmoe.frontend.adapter import NcclGroupChannel

import torch

from torch import Tensor

_tp_model_config: ModelConfig = None
_channel: NcclGroupChannel = None

def set_tensor_model_parallel_config(model_config: ModelConfig):
    global _tp_model_config
    _tp_model_config = model_config
    
def set_tensor_model_parallel_channel(channel: NcclGroupChannel):
    global _channel
    _channel = channel

def get_tensor_model_parallel_rank() -> int:
    return _tp_model_config.rank

def get_tensor_model_parallel_world_size() -> int:
    return _tp_model_config.tp_size

def tensor_model_parallel_all_reduce(tensor: Tensor) -> Tensor:
    if _tp_model_config.tp_size == 1:
        return tensor
    assert _channel is not None
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    _channel.all_reduce(tensor.data_ptr(), tensor.shape)
    return tensor

def tensor_model_parallel_all_gather(tensor: Tensor, dim: int = -1) -> Tensor:
    assert False, "No gather is needed"
    if _tp_model_config.tp_size == 1:
        return tensor
    assert _channel is not None
    assert dim == -1
    _channel.all_gather(tensor.data_ptr(), tensor.shape, dim)
    # FIXME(hogura|20241106): mocking tensor parallelism; remove this after integrating ATEN
    output = torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * _tp_model_config.tp_size),
                         dtype=tensor.dtype, device=tensor.device)
    return output