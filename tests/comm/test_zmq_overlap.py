import ray
import os

def func(rank: int):
    import torch
    from disagmoe_c import test_zmq_overlap, set_hosts, recorder_create, recorder_output
    from disagmoe.frontend.datatypes import TraceContext
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
                    dir_name="/home/hogura1999/DisagMoE/reports/zmq_overlap/", 
                    worker_name=f"worker-{rank}",
                    use_gzip=True,))
    recorder_create()
    # profiler.start()
    test_zmq_overlap(rank)
    # profiler.stop()
    output = recorder_output()
    result = {}
    for key in output:
        result[key] = [TraceContext.from_c(c) for c in output[key]]
    return result

def main():
    ray.init("auto")
    env = {
        "runtime_env": {
            # "CUDA_LAUNCH_BLOCKING": "1",
        }
    }
    tasks = [
        ray.remote(func).options(num_gpus=1, runtime_env=env).remote(0),
        ray.remote(func).options(num_gpus=1, runtime_env=env).remote(1),
        ray.remote(func).options(num_gpus=0, runtime_env=env).remote(82)
    ]
    outputs = ray.get(tasks)
    events = []
    ms_to_us = lambda ms: ms * 1000
    for pid, p_traces in enumerate(outputs):
        for tid, t_traces in p_traces.items():
            print("outputing thread", tid)
            tid = tid % (1 << 32)
            for trace in t_traces:
                if "schedule" in trace.msg and ms_to_us(trace.t_dur) < 10:
                    continue
                events.append({
                    "name": trace.msg,
                    "cat": "trace",
                    "ph": "X",
                    "ts": ms_to_us(trace.t_start),
                    "dur": ms_to_us(trace.t_dur),
                    "pid": pid,
                    "tid": (tid * 10 + trace.track_id) % (1 << 31),
                })

    import gzip
    import json
    with gzip.open(f"trace_zmq_overlap.json.gz", "w") as f:
        f.write(json.dumps(events).encode("utf-8"))
        

main()