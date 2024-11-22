

import torch
import os
import torch.distributed as dist
import time

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def init_process():

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def benchmark_broadcast():
    
    shape = (256, 4096)
    rank = dist.get_rank()
    data = torch.randn(shape).to(f"cuda:{rank}")
    
    # warmup
    for i in range(2):
        dist.broadcast(data, src=0)
        
    dist.barrier()
    
    torch.cuda.synchronize()
    start = time.time()
    
    repeats = 50
    
    for i in range(repeats):
        dist.broadcast(data, src=0)
    
    torch.cuda.synchronize()
    duration = (time.time() - start) / repeats
    print(f"rank {rank}, Broadcast Latency: {duration*1000*1000:.1f} us")

if __name__ == "__main__":
    
    init_process()
    benchmark_broadcast()
    