import ray

from disagmoe.utils.utils import get_nccl_unique_id

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        pass
    
    def run(self, rank, ranks, uid):
        import torch
        from disagmoe_c import test_op_overlap
        test_op_overlap(rank, ranks, uid)
        
def main():
    ray.init("auto")
    n = 2
    workers = [Worker.remote() for _ in range(n)]
    uid = get_nccl_unique_id()
    ray.get([w.run.remote(i, range(n), uid) for i, w in enumerate(workers)])

main()