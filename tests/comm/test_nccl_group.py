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
        
    def test_nccl(self):
        from disagmoe_c import test_nccl_group as test_nccl
        test_nccl(self.rank, self.ranks, self.uid)


def test_nccl_group():
    ray.init("auto")
    n = 4
    workers = [Worker.remote() for _ in range(n)]
    uid = get_nccl_unique_id()
    ray.get([w.setup.remote(i, list(range(n)), uid) for i, w in enumerate(workers)])
    ray.get([w.test_nccl.remote() for w in workers])
    
test_nccl_group()