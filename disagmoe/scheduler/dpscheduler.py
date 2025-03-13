from disagmoe.utils.logger import get_logger

from dataclasses import dataclass
from typing import List, Dict, override, Callable, Tuple
from functools import partial
from queue import Queue

import asyncio

@dataclass
class RequestItem:
    func: Callable
    req_id: int
    args: Tuple

class DPScheduler:
    
    def __init__(self, dp_size: int, seq_len: int, block_size: int):
        self.dp_size = dp_size
        self.kv_cache_stats = [0 for i in range(dp_size)]
        self.seq_len = seq_len
        self.block_size = block_size
        self.delta = (seq_len + block_size - 1) // block_size
        self.seq_ranks = {}
        
        self.end_flag = False
        self.end_event = asyncio.Event()
        self.sch_event = asyncio.Event()
        self.waiting_queue = asyncio.Queue()
        
        self._logger = get_logger("DPScheduler")
        
    def start(self, stats: Dict[int, int]):
        self._logger.info(f"Start with stats {stats}")
        self.init_kv_cache_stats(stats)
        asyncio.get_running_loop().create_task(self.waiting_loop())
        
    async def terminate(self):
        self.end_flag = True
        self.end_event.set()
    
    def put_request(self, func: Callable, req_id: int, *args):
        self.waiting_queue.put_nowait(RequestItem(func, req_id, args))
        self._logger.warning(f"Waiting queue put a request {req_id}, currently waiting list size {self.waiting_queue.qsize()}")
    
    async def waiting_loop(self):
        while not self.end_flag:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self.waiting_queue.get()), 
                    asyncio.create_task(self.end_event.wait())
                ],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if self.end_event.is_set():
                self._logger.warning("Waiting loop terminated")
                break
            
            for f in pending:
                f.cancel()
            
            request_item: RequestItem = done.pop().result()
            rank = self.schedule([request_item.req_id])[0]
            
            if rank < 0:
                await self.sch_event.wait()
                self.sch_event.clear()
                rank = self.schedule([request_item.req_id])[0]
                assert rank >= 0
            
            self._logger.warning(f"Waiting queue pop a request, assign {request_item.req_id} with rank {rank}, current waiting list size {self.waiting_queue.qsize()}")
            
            # submit the request
            request_item.func(request_item.req_id, *request_item.args, rank)
    
    def init_kv_cache_stats(self, stats: Dict[int, int]):
        for rank, num_blocks in stats.items():
            self.kv_cache_stats[rank] = num_blocks
        print(f"Init cache stats {self.kv_cache_stats}")
    
    def add_seq(self, seq_id: int, rank: int):
        self.kv_cache_stats[rank] -= self.delta
        self.seq_ranks[seq_id] = rank
        
    def del_seq(self, seq_id: int):
        rank = self.seq_ranks[seq_id]
        self.seq_ranks.pop(seq_id)
        self.kv_cache_stats[rank] += self.delta
        if len(self.sch_event._waiters) > 0:
            self.sch_event.set()
        # self._logger.debug(f"Delete seq {seq_id}, rank {rank}, current cache stats {self.kv_cache_stats}")
        
    def _schedule(self) -> int:
        raise NotImplementedError()
    
    def schedule(self, req_ids: List[int]) -> List[int]:
        ranks = []
        for i in req_ids:
            rank = self._schedule()
            ranks.append(rank)
            if rank >= 0:
                self.add_seq(i, rank)
        return ranks

class DPSchedulerMax(DPScheduler):
    
    @override
    def _schedule(self) -> int:
        stat = 0
        rank = -1
        for i, num_blocks in enumerate(self.kv_cache_stats):
            if num_blocks > stat:
                stat = num_blocks
                rank = i
        # self._logger.debug(f"Schedule rank {rank} with cache stat {stat}, all: {self.kv_cache_stats}")
        if stat < self.delta:
            return -1
        else:
            return rank

class DPSChedulerRR(DPScheduler):
    
    def __init__(self, dp_size: int, seq_len: int, block_size: int):
        super().__init__(dp_size, seq_len, block_size)
        self.cur_rank = 0
    
    @override
    def _schedule(self) -> int:
        rank = self.cur_rank
        self.cur_rank = (self.cur_rank + 1) % self.dp_size
        return rank

_clses = {
    "RR": DPSChedulerRR,
    "max": DPSchedulerMax,
}

def get_dp_scheduler(dp_size: int, seq_len: int, block_size: int, policy: str) -> DPScheduler:
    cls = _clses[policy]
    return cls(dp_size, seq_len, block_size)
