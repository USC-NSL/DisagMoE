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

def nccl_sender(world_size, rank):
    iterations = 20
    total_elapse = 0
    t = torch.rand(shape, dtype=torch.bfloat16)
    target = 1 - rank
    
    dist.barrier()
    
    for _ in range(10):
        dist.send(t, target)
    
    dist.barrier()
    
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=2,
                warmup=3,
                active=15
            ),
            on_trace_ready=tensorboard_trace_handler("remote_sender")
    ) as prof:
        for i in range(iterations):
            # dist.barrier()
            
            torch.cuda.synchronize(device)
            start = time.time()
            dist.send(t, target)
            torch.cuda.synchronize(device)
            total_elapse += time.time() - start
            # print(f"sender time {start * (10 ** 6):.1f} us")
            prof.step()
            
    elapse = total_elapse / iterations * (10**6)
    
    print(f"NCCL sender Latency: {elapse} us")


def zmq_send(world_size, rank):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")   
    
    t = torch.rand(shape, device='cpu')
    
    def run(iterations):
        rec = []
        elapse = 0
        
        for i in range(iterations):
            send_time = time.time()
            p = pickle.dumps(t)
            socket.send(p)
            elapse += time.time() - send_time
            rec.append(send_time)
            
        # print(f"sender {rec}")
        print(f"zmq sender {elapse / iterations * (10 ** 6):.1f} us")
        
    dist.barrier()
    run(10)
    dist.barrier()
    run(20)
        

uuid = get_nccl_unique_id()

metadata = Metadata(list(shape))

def cpp_nccl_send(world_size, rank):

    other = 1 - rank
    c = create_channel_py_map(rank, other, {other: uuid})
    instantiate_channels([c])
    t = torch.rand(shape, dtype=torch.float16)
    
    dist.barrier()
    
    for _ in range(10):
        c.send(t.data_ptr(), metadata)
    
    dist.barrier()
    
    iterations = 20
    
    elapse = 0
    
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=2,
                warmup=3,
                active=15
            ),
            on_trace_ready=tensorboard_trace_handler("remote_cpp_sender")
    ) as prof:
        for i in range(iterations):
            torch.cuda.synchronize(device)
            start = time.time()
            
            c.send(t.data_ptr(), metadata)
            
            torch.cuda.synchronize(device)
            end = time.time()
            elapse += end - start
            # print(f"sender time {start * (10 ** 6):.1f}")
            prof.step()
        
        
    print(f"sender {elapse / iterations * (10 ** 6):.1f} us")
    

def zmq_nccl_send(world_size, rank):
    
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")   
    
    device = f"cuda:{rank}" # 0
    torch.set_default_device(device)
    init_torch_dist(world_size, rank=rank)
    
    other = 1 - rank
    c = create_channel_py_map(rank, other, {other: uuid})
    instantiate_channels([c])
    t = torch.rand(shape, dtype=torch.float16)
    d = [0] * 1024 * 5
    
    def send_once():
        p = pickle.dumps(d)
        socket.send(p)
        c.send(t.data_ptr(), metadata)
        
    dist.barrier()
    
    for _ in range(10):
        send_once()
    
    dist.barrier()
    
    iterations = 20
    elapse = 0
    
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        
        send_once()
        
        torch.cuda.synchronize()
        end = time.time()
        elapse += end - start
        
    print(f"zmq_nccl sender {elapse / iterations * (10 ** 6):.1f} us")
    
    
    
init_torch_dist(2, 0)

device = "cuda:0"
torch.set_default_device(device)
    
    
nccl_sender(2, 0)

zmq_send(2, 0)
