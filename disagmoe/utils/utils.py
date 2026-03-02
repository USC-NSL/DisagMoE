import os
import time
import torch
import ctypes
import socket
import subprocess
import re

from disagmoe.utils.logger import get_logger, new_logger

from torch import Tensor
from typing import List, Tuple, Dict, Union, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from disagmoe_c import range_push, range_pop
except:
    from torch.cuda.nvtx import range_push, range_pop
    
def get_nccl_unique_id():
    from torch.cuda.nccl import unique_id
    return unique_id()

def get_nccl_url_from_uid(uid):
    h = 0
    for i in uid:
        h = (h * 256 + i) % 10007
    print("hash result:", h)
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if master_addr is None or master_port is None:
        raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set")
    return f"{master_addr}:{int(master_port) + h}"

class Counter:

    def __init__(self, start: int = 0, end: int = 2_000_000_000, step: int = 1) -> None:
        self.start = start
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
        self.counter = self.start
    
def get_ip(host_ifname: str = ""):
    # adpated from VLLM: https://github.com/vllm-project/vllm/blob/v0.6.0/vllm/utils.py#L484
    host_ip = os.environ.get("HOST_IP", None)
    if host_ip:
        return host_ip

    if host_ifname:
        try:
            out = subprocess.check_output(
                ["ip", "-o", "-4", "addr", "show", "dev", host_ifname],
                text=True,
            )
            m = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/", out)
            if m:
                return m.group(1)
        except Exception:
            pass

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

    new_logger("utils").warning(
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
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()


class CudaRangeEvent:
    
    def __init__(self, enable_timing: bool = False):
        self._start = torch.cuda.Event(enable_timing=enable_timing)
        self._end = torch.cuda.Event(enable_timing=enable_timing)
    
    def start(self):
        self._start.record(torch.cuda.current_stream())
    
    def end(self):
        self._end.record(torch.cuda.current_stream())
        
    def timing(self):
        return self._start.elapsed_time(self._end)

def make_seqlens_cuda_tensor(lens: Union[List[int], Tensor]) -> Optional[Tensor]:
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

def make_seqlens_list(lens: Union[List[int], Tensor], dst=None) -> Optional[List[int]]:
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
    internal_layer_id: int
    pool_snapshot: Dict[int, int]
    
    thread_id: int = -1
    process_id: int = -1
    

def time_ms():
    return time.time() * 1000

class Timer:
    
    def __init__(self):
        self.timers = {}
        
    def start(self, name):
        self.timers[name] = time.time_ns()
        
    def stop(self, name):
        start = self.timers[name]
        cost_ms = (time.time_ns() - start) / 1e6
        self.timers[name] = cost_ms
        return cost_ms
    
    def get(self, name):
        return self.timers.get(name)
    
    def reset(self):
        self.timers.clear()
    
    @contextmanager
    def range(self, name):
        self.start(name)
        yield
        self.stop(name)
        
def _log_memory_usage(prefix: str = ""):
    free_memory, total_memory = torch.cuda.mem_get_info()
    get_logger().info(f"{prefix} CUDA free memory: {free_memory / (1024 ** 3):.2f} GB, "\
                        f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
    
def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


enable_sync_event_timeout = False

def sync_event_timeout(event: torch.cuda.Event, timeout: float = 10.0):
    if enable_sync_event_timeout:
        timed = 0
        timed = 50 * (10 ** (-6)) # 50 us
        while event.query() == False:
            time.sleep(timed)
            timed += timed
            if timed > timeout:
                raise TimeoutError("Timeout waiting for sync event")
    event.synchronize()
