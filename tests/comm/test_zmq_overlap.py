import ray
import os

def func(rank: int):
    import torch
    from disagmoe_c import test_zmq_overlap, set_hosts
    set_hosts(os.getpid(), {
        0: "0.0.0.0",
        1: "0.0.0.0",
        82: "0.0.0.0"
    })
    test_zmq_overlap(rank)

def main():
    ray.init("auto")
    tasks = [
        ray.remote(func).options(num_gpus=1).remote(0),
        ray.remote(func).options(num_gpus=1).remote(1),
        ray.remote(func).options(num_gpus=0).remote(82)   
    ]
    ray.get(tasks)

main()