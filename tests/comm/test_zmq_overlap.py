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
    profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name="./reports/", 
                    worker_name=f"engine-{rank}",
                    use_gzip=True,))
    profiler.start()
    test_zmq_overlap(rank)
    profiler.stop()

def main():
    ray.init("auto")
    tasks = [
        ray.remote(func).options(num_gpus=1).remote(0),
        ray.remote(func).options(num_gpus=1).remote(1),
        ray.remote(func).options(num_gpus=0).remote(82)
    ]
    ray.get(tasks)
    ray.shutdown()

main()