import os
import torch
import ctypes
import socket

from disagmoe.utils.logger import get_logger

from torch import Tensor
from typing import List, Tuple, Dict

def tensor_as_buf(buf: int, shape: List[int], dtype = torch.float16) -> Tensor:
    # TODO(hogura|20241003): change c_int16 to a dynamic type
    data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_int16))
    tensor = torch.frombuffer(data, dtype=dtype)
    print("received tensor:", tensor)
    return torch.ones(shape).cuda()
    # return tensor.view(*shape)
    
def get_nccl_unique_id():
    from torch.cuda.nccl import unique_id
    return unique_id()
    
class Counter:

    def __init__(self, start: int = 0, end: int = 1e9, step: int = 1) -> None:
        self.counter = start
        self.end = end
        self.step = step

    def __next__(self) -> int:
        i = self.counter
        self.counter += self.step
        if self.counter >= self.end:
            self.counter = 0
        return i

    def reset(self) -> None:
        self.counter = 0
    
def get_ip():
    # adpated from VLLM: https://github.com/vllm-project/vllm/blob/v0.6.0/vllm/utils.py#L484
    host_ip = os.environ.get("HOST_IP", None)
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    get_logger("utils").warning(
        "Failed to get the IP address, using 0.0.0.0 by default."
        " The value can be set by the environment variable",
        " `HOST_IP`.",
        stacklevel=2)
    return "0.0.0.0"