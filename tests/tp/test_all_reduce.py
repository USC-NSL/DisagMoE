import os
import ray
import torch
import torch.distributed as dist

@ray.remote(num_gpus=1)
def run(rank, size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "26500"
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    dist.init_process_group(backend="nccl", world_size=size, rank=rank)
    
    a = torch.randn([256, 4096])
    dist.all_reduce(a)
    dist.barrier()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    dist.all_reduce(a)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    return elapsed

ray.init()
n = 4
results = ray.get([run.remote(i, n) for i in range(n)])
print(results)