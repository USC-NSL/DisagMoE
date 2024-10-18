import torch
import zmq
import torch.distributed as dist
import time
from disagmoe_c import *
import torch.multiprocessing as mp
import os
import pickle
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
from disagmoe.utils.utils import get_nccl_unique_id

shape = (512, 4096) # 4MB

os.environ['MASTER_ADDR'] = '127.0.0.1'  # or the IP of your master node
os.environ['MASTER_PORT'] = '29500'  # choose a free port

def init_torch_dist(world_size, rank):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def nccl_recver(rank):
    iterations = 20
    
    t = torch.empty(shape, dtype=torch.bfloat16)
    
    total_elapse = 0
    source = 1 - rank
    
    dist.barrier()
    
    for _ in range(10):
        dist.recv(t, source)
        
    dist.barrier()
    
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=3,
                active=15
            ),
            on_trace_ready=tensorboard_trace_handler("recver")
    ) as prof:
        for i in range(iterations):
            # dist.barrier()
            
            torch.cuda.synchronize(device)
            start = time.time()
            
            dist.recv(t, source)
            torch.cuda.synchronize(device)
            end = time.time() 
            total_elapse += end - start
            # print(f"recver time {end * (10 ** 6):.1f} us")
            
            prof.step()
    
    elapse = total_elapse / iterations * (10**6)
    print(f"NCCL recver Latency: {elapse} us")

def zmq_recv(rank):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")
    
    def run(iterations):
        rec = []
        elapse = 0
        
        for i in range(iterations):
            start = time.time()
            p = socket.recv()
            t = pickle.loads(p)
            recv_time = time.time()
            elapse += recv_time - start
            rec.append(recv_time)
        # print(f"recver {rec}")
        print(f"recver {elapse / iterations * (10 ** 6):.1f} us")
        
    dist.barrier()
    run(10)
    dist.barrier()
    run(20)


uuid = get_nccl_unique_id()

metadata = Metadata(list(shape))

def cpp_nccl_recv(rank):

    
    other = 1 - rank
    c = create_channel_py_map(rank, other, {other: uuid})
    instantiate_channels([c])
    t = torch.empty(shape, dtype=torch.float16)
    
    dist.barrier()
    
    for _ in range(10):
        c.recv(t.data_ptr(), metadata)
        
    iterations = 20
    dist.barrier()
    
    elapse = 0
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=2,
                warmup=3,
                active=15
            ),
            on_trace_ready=tensorboard_trace_handler("cpp_recver")
    ) as prof:
        for i in range(iterations):
            torch.cuda.synchronize(device)
            start = time.time()
            
            c.recv(t.data_ptr(), metadata)
            
            torch.cuda.synchronize(device)
            end = time.time()
            # print(f"recver time {end * (10 ** 6):.1f}")
            elapse += end - start
            prof.step()
    
    print(f"recver {elapse / iterations * (10 ** 6):.1f} us")
    
def zmq_nccl_recv(world_size, rank):
    
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")
    
    device = f"cuda:{rank}" # 1
    
    other = 1 - rank
    c = create_channel_py_map(rank, other, {other: uuid})
    instantiate_channels([c])
    t = torch.empty(shape, dtype=torch.float16)
    
    def recv_once():
        p = socket.recv()
        d = pickle.loads(p)
        c.recv(t.data_ptr(), metadata)
        
    dist.barrier()
    
    for _ in range(10):
        recv_once()
        
    dist.barrier()
    
    iterations = 20
    elapse = 0
    
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        
        recv_once()
        
        torch.cuda.synchronize()
        end = time.time()
        elapse += end - start
    
    print(f"zmq_nccl recver {elapse / iterations * (10 ** 6):.1f} us")
    
    
init_torch_dist(2, 1)

device = "cuda:0"
torch.set_default_device(device)
    
nccl_recver(1)

zmq_recv(1)