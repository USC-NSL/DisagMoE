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
    # context = zmq.Context()
    # socket = context.socket(zmq.REQ)
    # socket.connect("tcp://localhost:5555")
    device = f"cuda:{rank}"
    torch.set_default_device(device)
    
    init_torch_dist(world_size, rank=rank)
    
    iterations = 5
    total_elapse = 0
    t = torch.rand(shape, dtype=torch.bfloat16)
    target = 1 - rank
    
    for _ in range(10):
        dist.send(t, target)
    
    dist.barrier()
    
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=1,
                warmup=2,
                active=2
            ),
            on_trace_ready=tensorboard_trace_handler("sender")
    ) as prof:
        for i in range(iterations):
            # dist.barrier()
            
            torch.cuda.synchronize(device)
            start = time.time()
            dist.send(t, target)
            torch.cuda.synchronize(device)
            total_elapse += time.time() - start
            print(f"sender time {start * (10 ** 6):.1f} us")
            prof.step()
            
    elapse = total_elapse / iterations * (10**6)
    
    print(f"NCCL sender Latency: {elapse} us")

    
def nccl_recver(world_size, rank):
    # context = zmq.Context()
    # socket = context.socket(zmq.REP)
    # socket.bind("tcp://*:5555")
    
    iterations = 5
    device = f"cuda:{rank}"
    torch.set_default_device(device)
    
    init_torch_dist(world_size, rank)
    t = torch.empty(shape, dtype=torch.bfloat16)
    
    total_elapse = 0
    source = 1 - rank
    
    
    for _ in range(10):
        dist.recv(t, source)
        
    dist.barrier()
    
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=2,
                active=2
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
            print(f"recver time {end * (10 ** 6):.1f} us")
            
            prof.step()
    
    elapse = total_elapse / iterations * (10**6)
    print(f"NCCL recver Latency: {elapse} us")

    

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
            
        print(f"sender {rec}")
        print(f"sender {elapse / iterations * (10 ** 6):.1f} us")
    
    run(20)
        

def zmq_recv(world_size, rank):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")
    
    elapse = 0
    
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
        print(f"recver {rec}")
        print(f"recver {elapse / iterations * (10 ** 6):.1f} us")
        
    run(20)


uuid = get_nccl_unique_id()
print(f"uuid {uuid}")

metadata = Metadata(list(shape))

def cpp_nccl_send(world_size, rank):
    device = f"cuda:{rank}" # 0
    torch.set_default_device(device)
    init_torch_dist(world_size, rank=rank)
    
    other = 1 - rank
    c = create_channel_py_map(rank, other, {other: uuid})
    instantiate_channels([c])
    t = torch.rand(shape, dtype=torch.float16)
    for _ in range(10):
        c.send(t.data_ptr(), metadata)
    
    dist.barrier()
    
    iterations = 20
    
    elapse = 0
    for i in range(iterations):
        torch.cuda.synchronize(device)
        start = time.time()
        
        c.send(t.data_ptr(), metadata)
        
        torch.cuda.synchronize(device)
        end = time.time()
        elapse += end - start
        print(f"sender time {start * (10 ** 6):.1f}")
        
        
    print(f"sender {elapse / iterations * (10 ** 6):.1f} us")
    

def cpp_nccl_recv(world_size, rank):
    device = f"cuda:{rank}" # 1
    torch.set_default_device(device)
    init_torch_dist(world_size, rank=rank)
    
    other = 1 - rank
    c = create_channel_py_map(rank, other, {other: uuid})
    instantiate_channels([c])
    t = torch.empty(shape, dtype=torch.float16)
    
    for _ in range(10):
        c.recv(t.data_ptr(), metadata)
        
    iterations = 20
    dist.barrier()
    
    
    elapse = 0
    for i in range(iterations):
        torch.cuda.synchronize(device)
        start = time.time()
        
        c.recv(t.data_ptr(), metadata)
        
        torch.cuda.synchronize(device)
        end = time.time()
        print(f"recver time {end * (10 ** 6):.1f}")
        elapse += end - start
    
    print(f"recver {elapse / iterations * (10 ** 6):.1f} us")
    
        
    
    

world_size = 2
nccl_funcs = [nccl_sender, nccl_recver]
zmq_funcs = [zmq_send, zmq_recv]
cpp_nccl_funcs = [cpp_nccl_send, cpp_nccl_recv]

def test_latency(funcs):
    procs = []

    for i in range(world_size):
        p = mp.Process(target=funcs[i], args=(world_size, i,))
        p.start()
        procs.append(p)
        
    for p in procs:
        p.join()
        
test_latency(nccl_funcs)

# test_latency(zmq_funcs)


# cpp_nccl should not define enable_ray
test_latency(cpp_nccl_funcs)