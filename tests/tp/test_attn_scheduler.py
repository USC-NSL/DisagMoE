import ray
import torch

from typing import List
from disagmoe.utils.utils import get_nccl_unique_id

@ray.remote(num_gpus=1)
class Worker:
    
    def __init__(self):
        pass
    
    def setup(self, rank: int, ranks: List[int], uid):
        self.rank = rank
        self.uid = uid
        self.ranks = ranks
        torch.set_default_device("cuda:0")
        
    def test(self):
        from disagmoe_c import test_parallel_attn_scheduler
        test_parallel_attn_scheduler(self.rank, self.ranks, self.uid)
        print("passed")


def test_attn_scheduler():
    ray.init("auto")
    n = 4
    workers = [Worker.remote() for _ in range(n)]
    uid = get_nccl_unique_id()
    ray.get([w.setup.remote(i, list(range(n)), uid) for i, w in enumerate(workers)])
    ray.get([w.test.remote() for w in workers])
    
test_attn_scheduler()