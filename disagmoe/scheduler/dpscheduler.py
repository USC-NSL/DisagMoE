from typing import List, Dict, override

class DPScheduler:
    
    def __init__(self, dp_size: int, seq_len: int, block_size: int):
        self.dp_size = dp_size
        self.kv_cache_stats = [0 for i in range(dp_size)]
        self.seq_len = seq_len
        self.block_size = block_size
        self.delta = (seq_len + block_size - 1) // block_size
        self.seq_ranks = {}
    
    def init_kv_cache_stats(self, stats: Dict[int, int]):
        for rank, num_blocks in stats.items():
            self.kv_cache_stats[rank] = num_blocks
    
    def add_seq(self, seq_id: int, rank: int):
        self.kv_cache_stats[rank] += self.delta
        self.seq_ranks[seq_id] = rank
        
    def del_seq(self, seq_id: int):
        rank = self.seq_ranks[seq_id]
        self.seq_ranks.pop(seq_id)
        self.kv_cache_stats[rank] -= self.delta
        
    def _schedule(self) -> int:
        raise NotImplementedError()
    
    def schedule(self, req_ids: List[int]) -> List[int]:
        ranks = []
        for i in req_ids:
            rank = self._schedule()
            ranks.append(rank)
            self.add_seq(i, rank)
        return ranks

class DPSchedulerMax(DPScheduler):
    
    @override
    def _schedule(self) -> int:
        stat = self.kv_cache_stats[0]
        rank = 0
        for i, num_blocks in enumerate(self.kv_cache_stats[1: ]):
            if num_blocks > stat:
                stat = num_blocks
                rank = i
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
