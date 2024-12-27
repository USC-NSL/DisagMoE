import os
import torch
import ctypes
import socket

from disagmoe.utils.logger import get_logger

from torch import Tensor
from typing import List, Tuple, Dict, Union
from contextlib import contextmanager
from dataclasses import dataclass

def get_nccl_unique_id():
    from torch.cuda.nccl import unique_id
    return unique_id()

def get_nccl_url_from_uid(uid):
    h = 0
    for i in uid:
        h = (h * 256 + i) % 10007
    print("hash result:", h)
    return f"{os.getenv("MASTER_ADDR")}:{int(os.getenv("MASTER_PORT")) + h}"

class Counter:

    def __init__(self, start: int = 0, end: int = 2e9, step: int = 1) -> None:
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


@contextmanager
def nvtx_range_cuda(msg, *args, **kwargs):
    """ 
    From vLLM: https://github.com/vllm-project/vllm/blob/7abba39ee64c1e2c84f48d7c38b2cd1c24bb0ebb/vllm/spec_decode/util.py#L238
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    If running with cuda graphs, you must enable nsys cuda graph profiling.

    Arguments:
        msg (string): message to associate with the range
    """
    torch.cuda.nvtx.range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def nvtx_range(msg, *args, **kwargs):
    from disagmoe_c import range_push, range_pop
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()


def make_seqlens_cuda_tensor(lens: Union[List[int], Tensor]) -> Tensor:
    if isinstance(lens, Tensor):
        lens = lens.view(-1).tolist()
    if len(lens) == 0:
        return None
    seqlen = [0]
    for l in lens:
        seqlen.append(seqlen[-1] + l)
    result = torch.tensor(seqlen, dtype=torch.int32, device="cuda")
    return result

def get_graph_batch_size(batch_size: int, graph_batch_sizes: List[int]) -> Tuple[int, int]:
    for i, size in enumerate(graph_batch_sizes):
        if size >= batch_size:
            return i, size
    assert False, f"No available graph for batch size={batch_size}"

def make_seqlens_list(lens: Union[List[int], Tensor], dst=None) -> List[int]:
    if isinstance(lens, Tensor):
        lens = lens.view(-1).tolist()
    n = len(lens)
    
    if n == 0:
        return None
    
    if dst is None:
        dst = [0] * (n + 1)
    else:    
        assert len(dst) == n + 1
    
    dst[0] = 0
    for i in range(n):
        dst[i+1] = dst[i] + lens[i]
    return dst


@dataclass
class StepInfo:
    start_timestamp_ms: float
    end_timestamp_ms: float
    batch_size: int
    layer_id: int
    pool_snapshot: Dict[int, int]
    
    thread_id: int = -1
    process_id: int = -1