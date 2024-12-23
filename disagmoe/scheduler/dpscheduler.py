from typing import List, Dict, override

class DPScheduler:
    
    def __init__(self, dp_size: int, seq_len: int, block_size: int):
        self.dp_size = dp_size
        self.kv_cache_stats = [0 for i in range(dp_size)]
        self.seq_len = seq_len
        self.block_size = block_size
        self.delta = (seq_len + block_size - 1) // block_size
    
    def init_kv_cache_stats(self, stats: Dict[int, int]):
        for rank, num_blocks in stats.items():
            self.kv_cache_stats[rank] = num_blocks
    
    def add_seq(self, rank: int):
        self.kv_cache_stats[rank] += self.delta
        
    def del_seq(self, rank: int):
        self.kv_cache_stats[rank] -= self.delta
    
    def schedule(self) -> int:
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
    def schedule(self) -> int:
        rank = self.cur_rank
        self.cur_rank = (self.cur_rank + 1) % self.dp_size
        return rank

_clses = {
    "RR": DPSChedulerRR,
    "max": DPScheduler
}

def get_dp_scheduler(dp_size: int, seq_len: int, block_size: int, policy: str) -> DPScheduler:
    cls = _clses[policy]
    return cls(dp_size, seq_len, block_size)
