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
    seq_len: int
    prefill_len: int
    output_len: int

class DPScheduler:
    
    def __init__(self, dp_size: int, max_output_len: int, block_size: int):
        self.dp_size = dp_size
        self.kv_cache_stats = [0 for i in range(dp_size)]
        self.max_output_len = max_output_len
        self.block_size = block_size
        self.seq_ranks = {}
        self.seq_max_len = {}
        
        self._logger = get_logger("DPScheduler")
        self._loop_task: asyncio.Task = None
        
        self.reset()
        
    def reset(self):
        self.end_flag = False
        self.end_event = asyncio.Event()
        self.sch_event = asyncio.Event()
        self.waiting_queue = asyncio.Queue()
        
    def start(self, stats: Dict[int, int]):
        self._logger.info(f"Start with stats {stats}")
        self.init_kv_cache_stats(stats)
        self.reset()
        
        self._loop_task = asyncio.create_task(self.waiting_loop())
        self._log_task = asyncio.create_task(self.log_status())
        self._logger.warning("Created waiting loop")
        
    async def terminate(self):
        self.end_flag = True
        self.end_event.set()
        await self._loop_task
        
    async def log_status(self):
        while not self.end_flag:
            self._logger.info(f"Global DP scheduler: #running requests: {len(self.seq_ranks)}, #waiting requests: {self.waiting_queue.qsize()}")
            await asyncio.sleep(10)
        
    def put_request(self, func: Callable, req_id: int, seq_len: int, prefill_len: int, output_len: int):
        self.waiting_queue.put_nowait(RequestItem(func, req_id, seq_len, prefill_len, output_len))
        
    def required_blocks(self, seq_len: int) -> int:
        return (seq_len + self.block_size - 1) // self.block_size
    
    async def waiting_loop(self):
        self._logger.warning("Waiting loop started")
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
            rank = self.schedule([request_item.req_id], [request_item.seq_len])[0]
            
            while rank < 0:
                await self.sch_event.wait()
                self.sch_event.clear()
                rank = self.schedule([request_item.req_id], [request_item.seq_len])[0]
            
            # self._logger.warning(f"Waiting queue pop a request, assign {request_item.req_id} with rank {rank}, current waiting list size {self.waiting_queue.qsize()}")
            
            # submit the request
            request_item.func(request_item.req_id, request_item.prefill_len, request_item.output_len, rank)
    
    def init_kv_cache_stats(self, stats: Dict[int, int]):
        for rank, num_blocks in stats.items():
            self.kv_cache_stats[rank] = num_blocks
        print(f"Init cache stats {self.kv_cache_stats}")
    
    def add_seq(self, seq_id: int, max_length: int, rank: int):
        self.seq_max_len[seq_id] = max_length
        required_blocks = self.required_blocks(max_length)
        self.kv_cache_stats[rank] -= required_blocks
        self.seq_ranks[seq_id] = rank
        
    def del_seq(self, seq_id: int):
        rank = self.seq_ranks[seq_id]
        self.seq_ranks.pop(seq_id)
        self.kv_cache_stats[rank] += self.required_blocks(self.seq_max_len[seq_id])
        self.seq_max_len.pop(seq_id)
        if len(self.sch_event._waiters) > 0:
            self.sch_event.set()
        # self._logger.info(f"Delete seq {seq_id}, rank {rank}, current cache stats {self.kv_cache_stats}")
        
    def _schedule(self, seq_len: int) -> int:
        raise NotImplementedError()
    
    def schedule(self, req_ids: List[int], seq_lens: List[int]) -> List[int]:
        ranks = []
        for req, seq_len in zip(req_ids, seq_lens):
            rank = self._schedule(seq_len)
            ranks.append(rank)
            if rank >= 0:
                self.add_seq(req, seq_len, rank)
        return ranks

class DPSchedulerMax(DPScheduler):
    
    @override
    def _schedule(self, seq_len: int) -> int:
        stat = 0
        rank = -1
        for i, num_blocks in enumerate(self.kv_cache_stats):
            if num_blocks > stat:
                stat = num_blocks
                rank = i
        required_blocks = self.required_blocks(seq_len)
        if stat < required_blocks:
            return -1
        else:
            return rank

class DPSChedulerRR(DPScheduler):
    
    def __init__(self, dp_size: int, seq_len: int, block_size: int):
        super().__init__(dp_size, seq_len, block_size)
        self.cur_rank = 0
    
    @override
    def _schedule(self, seq_len: int) -> int:
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
