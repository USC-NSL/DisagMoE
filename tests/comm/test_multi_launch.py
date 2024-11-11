import ray
import torch

from disagmoe.utils.utils import get_nccl_unique_id

@ray.remote(num_gpus=1)
class Worker:
    
    def __init__(self):
        pass
    
    def setup(self, rank: int, ranks: int, uids):
        self.rank = rank
        self.uids = uids
        self.ranks = ranks
        torch.set_default_device("cuda:0")
        
    def test_multi_launch(self):
        from disagmoe_c import test_multi_launch
        test_multi_launch(self.rank, self.ranks, self.uids)
        
def main():
    ray.init("auto")
    n = 4
    m = 6
    workers = [Worker.remote() for _ in range(n)]
    uids = [get_nccl_unique_id() for _ in range(m)]
    ray.get([w.setup.remote(i, range(n), uids) for i, w in enumerate(workers)])
    ray.get([w.test_multi_launch.remote() for w in workers])

main()