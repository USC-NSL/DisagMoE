from disagmoe.frontend.controller import init_controller, Controller, AsyncResult
from disagmoe.utils.placement import ModelPlacement, ClusterConfig, get_model_placement
from disagmoe.utils.utils import StepInfo
from disagmoe.utils.constants import *
from disagmoe.config import ModelConfig, CacheConfig, mixtral_config, SamplingConfig
from disagmoe.frontend.datatypes import SloStat
from typing import List
from argparse import ArgumentParser
from dataclasses import dataclass

import asyncio
import time
import tqdm
import numpy as np
import pandas as pd

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

master: Controller = None

@dataclass
class BenchmarkMetrics:
    
    e2e_duration: float
    req_throughput: float
    token_throughput: float
    
    req_latency_median_ms: float
    req_latency_p90_ms: float
    req_latency_p99_ms: float
    
    itl_latency_median_ms: float
    itl_latency_p90_ms: float
    itl_latency_p99_ms: float
    
    def __repr__(self):
        return f"Metrics: \n" \
                f"e2e_duration: {self.e2e_duration:.2f}s\n" \
                f"req_throughput: {self.req_throughput:.2f} req/s\n" \
                f"token_throughput: {self.token_throughput:.2f} tokens/s\n" \
                f"req_latency_median: {self.req_latency_median_ms:.2f}ms\n" \
                f"req_latency_p90: {self.req_latency_p90_ms:.2f}ms\n" \
                f"req_latency_p99: {self.req_latency_p99_ms:.2f}ms\n" \
                f"itl_latency_median: {self.itl_latency_median_ms:.2f}ms\n" \
                f"itl_latency_p90: {self.itl_latency_p90_ms:.2f}ms\n" \
                f"itl_latency_p99: {self.itl_latency_p99_ms:.2f}ms\n"

def launch(args):
    cluster_config = ClusterConfig(n_node=args.num_nodes, n_gpu=args.num_gpus,
                                id_tokenizer=tokenizer, 
                                id_sampler=sampler)

    model_config = mixtral_config
    model_config.num_layers = args.num_layers
    model_config.ep_size = args.ep_size
    model_config.tp_size = args.tp_size
    model_config.tp_enable_inter_group = False
    model_config.enable_cuda_graph = args.cuda_graph
    model_config.num_experts = args.num_experts
    model_config.dp_size = args.dp_size

    mp = get_model_placement(model_config, cluster_config, args.placement, 
                             step_attn=args.step_attn, step_expert=args.step_expert, 
                             zigzag_attn=args.zigzag_attn)
    # mp = get_model_placement(model_config, cluster_config, "interleave")

    global master

    master = init_controller(cluster_config.n_node, cluster_config.n_gpu, args.nsys)

    cache_config = CacheConfig(args.block_size, 0.9, 2, "auto",
                               num_reserved_blocks=RESERVED_BLOCKS)

    sampling_config = SamplingConfig(max_output_len=args.output_len)
    
    master.init_engine(mp, model_config, cache_config, sampling_config)
    
    if args.profile_dir is not None:
        master.start_profile(args.profile_dir)

    master.start_engine()
    
   
async def process_response(resp: AsyncResult, pbar):
    slo_stat = await resp.get()
    # print(f">>> Response received: {resp.req_id}, {slo_stat}")
    pbar.update(1)
    return slo_stat

def analyze_results(results: List[SloStat], duration: float):
    req_latency = []
    itls = []
    total_output_tokens = 0
    num_reqs = len(results)
    for result in results:
        total_output_tokens += len(result.t_tokens)
        itls.extend(result.t_tokens)
        req_latency.append(result.t_decode)
        
    
    return BenchmarkMetrics(
        e2e_duration=duration,
        req_throughput=num_reqs / duration,
        token_throughput=total_output_tokens / duration,
        
        req_latency_median_ms=np.median(req_latency) * 1000,
        req_latency_p90_ms=np.percentile(req_latency, 90) * 1000,
        req_latency_p99_ms=np.percentile(req_latency, 99) * 1000,
        
        itl_latency_median_ms=np.median(itls) * 1000,
        itl_latency_p90_ms=np.percentile(itls, 90) * 1000,
        itl_latency_p99_ms=np.percentile(itls, 99) * 1000,
    )

def analyze_batch_sizes(all_batch_sizes: List[List[int]]):
    for i, worker_batch_sizes in enumerate(all_batch_sizes):

        try:
            import matplotlib.pyplot as plt
            plt.figure()
            df = pd.DataFrame(worker_batch_sizes)
            plt.plot(df)
            plt.title(f"Worker {i} batch sizes with time")
            plt.savefig(f"worker_{i}_batch_sizes_with_time.png")
            plt.close()
            
            plt.figure()
            plt.hist(worker_batch_sizes, bins=32)
            plt.title(f"Worker {i} batch sizes")
            plt.savefig(f"worker_{i}_batch_sizes.png")
            plt.close()
            
        except:
            print("matplotlib not found, skipping plotting")
            

def generate_step_trace(step_stats: List[List[StepInfo]]):
    events = []
    
    def ms_to_us(ms):
        return ms * 1000
    
    for worker_id, worker_stats in enumerate(step_stats):
        for step_info in worker_stats:
            events.append({
                "name": f"layer {step_info.layer_id}, batch {step_info.batch_size}",
                "cat": "step",
                "ph": "X",
                "ts": ms_to_us(step_info.start_timestamp_ms),
                "dur": ms_to_us(step_info.end_timestamp_ms - step_info.start_timestamp_ms),
                "pid": 0,
                "tid": worker_id,
                "args": {
                    "pool_snapshot": f"{step_info.pool_snapshot}"
                }
            })
            
    with open("steps_trace.json", "w") as f:
        import json
        json.dump(events, f)

async def benchmark_serving(args):
    assert master is not None, "master is not initialized"
    assert args.input_len == 1, "supports only 1 token as input"
    
    master.start_polling_results()
    
    print(f"generating requests at rate {args.rate} s/req, in total {args.num_requests} requests")
    
    async def run_once():
        pbar = tqdm.tqdm(total=args.num_requests)
        benchmark_start_time = time.perf_counter()
        tasks = []
        for _ in range(args.num_requests):
            resp = master.put_single_request(args.input_len)
            tasks.append(asyncio.create_task(process_response(resp, pbar)))
            await asyncio.sleep(args.rate)
        
        results: List[SloStat] = await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - benchmark_start_time
        pbar.close()

        metrics = analyze_results(results, benchmark_duration)
        
        print(metrics)
    
    await run_once()
    
    step_stats = master.fetch_step_stats()
    
    master.stop_workers()
    
    generate_step_trace(step_stats)
    
    
def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("-r", "--rate", type=float, default=0, help="rate of incoming requests, seconds per request")
    parser.add_argument("-i", "--input-len", type=int, default=1, help="length of input sequence")
    parser.add_argument("-o", "--output-len", type=int, default=32, help="length of output sequence")
    parser.add_argument("-n", "--num-requests", type=int, default=1000, help="number of requests to generate")
    parser.add_argument("-p", "--profile-dir", type=str, default=None, help="directory to store torch profiler output")
    parser.add_argument("-c", "--cuda-graph", action="store_true", default=False, help="enable cuda graph")
    parser.add_argument("--nsys", action="store_true", help="enable nsys profiling")
    
    # model config
    parser.add_argument("-N", "--num-nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("-g", "--num-gpus", type=int, default=4, help="number of gpus per node")
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--ep-size", type=int, default=2, help="expert parallel size")
    parser.add_argument("--dp-size", type=int, default=1, help="data parallel size")
    parser.add_argument("-L", "--num-layers", type=int, default=32, help="number of layers")
    parser.add_argument("-E", "--num-experts", type=int, default=8, help="number of experts")
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS, help="number of blocks in cache; deprycated due to auto-num-blocks")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="block size in cache")
    
    # placement config
    parser.add_argument("--placement", type=str, default="pipeline", help="placement strategy")
    parser.add_argument("--zigzag-attn", action="store_true", default=True, help="enable zigzag attention placment")
    parser.add_argument("--step-attn", type=int, default=2, help="number of steps in attention placement")
    parser.add_argument("--step-expert", type=int, default=1, help="number of steps in expert placement")
    
    args = parser.parse_args()
    
    if (args.num_nodes * args.num_gpus) % (args.tp_size * args.step_attn + args.ep_size * args.step_expert) != 0:
        print("Warning: number of gpus is not divisible by the number of placement steps")
    
    assert args.ep_size <= args.num_experts, "expert parallel size must be smaller than number of experts"
    assert args.num_experts % args.ep_size == 0, "number of experts must be divisible by expert parallel size"
    
    if args.nsys:
        assert args.profile_dir is None, "cannot enable both nsys and torch profiler"
        
    return args

def main():
    args = get_args()
    
    launch(args)
    
    asyncio.run(benchmark_serving(args))
    
if __name__ == "__main__":
    main()