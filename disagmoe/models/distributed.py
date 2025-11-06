from disagmoe.config import ModelConfig
from disagmoe.utils.logger import new_logger

import torch
import torch.distributed as dist

from torch import Tensor

_logger = new_logger("dist")

_tp_model_config: ModelConfig = None

'''
Consider cleanup the unused tp code below
'''
def set_tensor_model_parallel_config(model_config: ModelConfig):
    global _tp_model_config
    _tp_model_config = model_config

def get_tensor_model_parallel_rank() -> int:
    return _tp_model_config.rank

def get_tensor_model_parallel_world_size() -> int:
    return _tp_model_config.tp_size

def tensor_model_parallel_all_reduce(tensor: Tensor) -> Tensor:
    if _tp_model_config.tp_size == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    dist.all_reduce(tensor)
    return tensor

def group_sync() -> None:
    if _tp_model_config.tp_size == 1:
        return
    dist.barrier()

def tensor_model_parallel_all_gather(tensor: Tensor, dim: int = -1) -> Tensor:
    assert False, "AllGather is not required currently"
    if _tp_model_config.tp_size == 1:
        return tensor
    assert _channel is not None
    assert dim == -1
    _channel.all_gather(tensor.data_ptr(), tensor.shape, dim)
    # FIXME(hogura|20241106): mocking tensor parallelism; remove this after integrating ATEN
    output = torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * _tp_model_config.tp_size),
                         dtype=tensor.dtype, device=tensor.device)
    return output