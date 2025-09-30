from disagmoe.config import ModelConfig
from disagmoe.frontend.adapter import NcclGroupChannel
from disagmoe.utils.logger import get_logger

import torch
import torch.distributed as dist

from torch import Tensor

_logger = get_logger("dist")

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
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if _tp_model_config.tp_enable_inter_group:
        assert _channel is not None
        _channel.all_reduce(tensor.data_ptr(), tensor.shape)
    else:
        dist.all_reduce(tensor)
    return tensor

def group_sync() -> None:
    if _tp_model_config.tp_size == 1:
        return
    if _tp_model_config.tp_enable_inter_group:
        assert _channel is not None
        _channel.all_reduce(torch.zeros(1).data_ptr(), (1,))
    else:
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