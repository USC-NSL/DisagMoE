from disagmoe.frontend.controller import init_controller, Controller, AsyncResult
from disagmoe.utils.placement import ModelPlacement, ClusterConfig, get_model_placement
from disagmoe.utils.utils import StepInfo
from disagmoe.utils.metrics import Metric
from disagmoe.utils.constants import *
from disagmoe.config import ModelConfig, CacheConfig, mixtral_config, SamplingConfig
from disagmoe.frontend.datatypes import SloStat, TraceContext
from typing import List, Dict, Tuple
from argparse import ArgumentParser
from dataclasses import dataclass, asdict

import gzip
import json
import asyncio
import time
import tqdm
import numpy as np
import pandas as pd
import os

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

master: Controller = None

@dataclass
class BenchmarkMetrics:
    e2e_duration: float = -1
    req_throughput: float = -1
    token_throughput: float = -1
    
    req_latency_median_ms: float = -1
    req_latency_p90_ms: float = -1
    req_latency_p99_ms: float = -1
    
    itl_latency_median_ms: float = -1
    itl_latency_p90_ms: float = -1
    itl_latency_p99_ms: float = -1
    
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

    def write_to_file(self, args):
        filename = args.file
        try:
            import pandas as pd
            if not os.path.exists(filename):
                df = pd.DataFrame(columns=["num_requests", "output_len",
                                           "step_attn", "DP_size", "max_batch_size_attn", 
                                           "step_expert", "EP_size", "max_batch_size_expert",
                                           "num_nodes", "num_gpus", "num_experts", "num_layers"] + list(self.__dict__.keys()))
            else:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filename)
                else:
                    df = pd.read_excel(filename)
            new_row = {
                "num_requests": args.num_requests,
                "output_len": args.output_len,
                "step_attn": args.step_attn,
                "DP_size": args.dp_size,
                "max_batch_size_attn": args.max_batch_size_attn,
                "step_expert": args.step_expert,
                "EP_size": args.ep_size,
                "max_batch_size_expert": args.max_batch_size_expert,
                "num_nodes": args.num_nodes,
                "num_gpus": args.num_gpus,
                "num_experts": args.num_experts,
                "num_layers": args.num_layers,
                **self.__dict__
            }
            df.loc[len(df)] = new_row
            if filename.endswith(".csv"):
                df.to_csv(filename, index=False)
            else:
                df.to_excel(filename, index=False)
        except Exception as e:
            print("Error: failed to write to file, with exception:", e)

def launch(args):
    cluster_config = ClusterConfig(n_node=args.num_nodes, n_gpu=args.num_gpus,
                                id_tokenizer=tokenizer, 
                                id_sampler=sampler)

    model_config = mixtral_config
    model_config.num_layers = args.num_layers
    model_config.ep_size = args.ep_size
    model_config.tp_size = args.tp_size
    model_config.tp_enable_inter_group = False
    model_config.enable_cuda_graph_attn = args.cuda_graph_attn
    model_config.enable_cuda_graph_expert = False
    model_config.enable_grouped_gemm = not args.serial_gemm
    model_config.num_experts = args.num_experts
    model_config.dp_size = args.dp_size
    model_config.max_batch_size_attn = args.max_batch_size_attn
    model_config.max_batch_size_expert = args.max_batch_size_expert
    model_config.graph_stride = args.graph_stride
    model_config.top_k = args.topk
    model_config.enable_trace = args.trace

    mp = get_model_placement(model_config, cluster_config, args.placement, 
                             step_attn=args.step_attn, step_expert=args.step_expert, 
                             zigzag_attn=args.zigzag_attn)
    # mp = get_model_placement(model_config, cluster_config, "interleave")

    global master

    master = init_controller(cluster_config.n_node, cluster_config.n_gpu, args.nsys)

    cache_config = CacheConfig(args.block_size, 0.9, 2, "auto",
                               num_gpu_blocks=args.num_blocks + RESERVED_BLOCKS if args.num_blocks else None, # default should be None
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
    try:
        import matplotlib.pyplot as plt 
    except:
        print("matplotlib not found, skipping plotting")
    
    for i, worker_batch_sizes in enumerate(all_batch_sizes):
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


def generate_step_trace(args,
                        step_stats: List[ Tuple[List[StepInfo], Dict[int, List[TraceContext]], Metric] ]):
    trace_name = f"trace_step-attn={args.step_attn}_dp-size={args.dp_size}_step-exp={args.step_expert}_ep-size={args.ep_size}" \
                 f"_layers={args.num_layers}_experts={args.num_experts}" \
                 f"_max-batch-attn={args.max_batch_size_attn}_max-batch-exp={args.max_batch_size_expert}"
    events = []
    
    def ms_to_us(ms):
        return ms * 1000
    
    metrics = {}
    
    for pid, (worker_stats, p_traces, metric) in enumerate(step_stats):
        for step_info in worker_stats:
            events.append({
                "name": f"layer {step_info.layer_id}, batch {step_info.batch_size}",
                "cat": "step",
                "ph": "X",
                "ts": ms_to_us(step_info.start_timestamp_ms),
                "dur": ms_to_us(step_info.end_timestamp_ms - step_info.start_timestamp_ms),
                "pid": pid,
                "tid": 0,
                "args": {
                    "pool_snapshot": f"{step_info.pool_snapshot}"
                }
            })
            
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
                
        metrics[pid] = asdict(metric)

    with gzip.open(f"{trace_name}.json.gz", "w") as f:
        f.write(json.dumps(events).encode("utf-8"))
        
    with open(f"metrics_{trace_name}.json", "w") as f:
        json.dump(metrics, f)


async def benchmark_serving(args):
    assert master is not None, "master is not initialized"
    
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

        metrics.write_to_file(args)
    
    await master.start_scheduler()
    await run_once()
    await master.stop_scheduler()
    
    master.stop_workers()
    
    if args.trace:
        step_stats = master.fetch_step_stats()
        generate_step_trace(args, step_stats)
    
def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("-r", "--rate", type=float, default=0, help="rate of incoming requests, seconds per request")
    parser.add_argument("-i", "--input-len", type=int, default=1, help="initial prefill length for each seqeunce")
    parser.add_argument("-o", "--output-len", type=int, default=32, help="length of output sequence")
    parser.add_argument("-n", "--num-requests", type=int, default=1000, help="number of requests to generate")
    parser.add_argument("-p", "--profile-dir", type=str, default=None, help="directory to store torch profiler output")
    parser.add_argument("-ca", "--cuda-graph-attn", action="store_true", default=False, help="enable cuda graph for attention")
    parser.add_argument("--serial-gemm", action="store_true", default=False, help="use serial gemm for experts")
    parser.add_argument("--nsys", action="store_true", help="enable nsys profiling")
    parser.add_argument("-f", "--file", type=str, default="reports/benchmark.xlsx", help="file to write benchmark results")
    parser.add_argument("--trace", action="store_true", default=False, help="generate trace")
    
    # model config
    parser.add_argument("-N", "--num-nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("-g", "--num-gpus", type=int, default=4, help="number of gpus per node")
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--ep-size", type=int, default=2, help="expert parallel size")
    parser.add_argument("--dp-size", type=int, default=1, help="data parallel size")
    parser.add_argument("-L", "--num-layers", type=int, default=32, help="number of layers")
    parser.add_argument("-E", "--num-experts", type=int, default=8, help="number of experts")
    parser.add_argument("-K", "--topk", type=int, default=1, help="top k")
    parser.add_argument("--num-blocks", type=int, default=None, help="number of blocks in cache; deprycated due to auto-num-blocks")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="block size in cache")
    parser.add_argument("--graph-stride", type=int, default=32, help="CUDA graph batch size stride")
    parser.add_argument("--max-batch-size-attn", type=int, default=256, help="max batch size for attention")
    parser.add_argument("--max-batch-size-expert", type=int, default=512, help="max batch size for experts")
    
    # placement config
    parser.add_argument("--placement", type=str, default="pipeline", help="placement strategy")
    parser.add_argument("--zigzag-attn", action="store_true", default=False, help="enable zigzag attention placment")
    parser.add_argument("--step-attn", type=int, default=2, help="number of steps in attention placement")
    parser.add_argument("--step-expert", type=int, default=1, help="number of steps in expert placement")
    
    args = parser.parse_args()
    
    assert args.num_experts >= args.topk, "number of experts must be greater than topk"
    
    if (args.num_nodes * args.num_gpus) % (args.tp_size * args.dp_size * args.step_attn + args.ep_size * args.step_expert) != 0:
        print("Warning: number of gpus is not divisible by the number of placement steps")
    
    assert args.ep_size <= args.num_experts, "expert parallel size must be smaller than number of experts"
    assert args.num_experts % args.ep_size == 0, "number of experts must be divisible by expert parallel size"
    
    if args.nsys:
        assert args.profile_dir is None, "cannot enable both nsys and torch profiler"
        
    return args

def main():
    args = get_args()
    
    # try:
    launch(args)
    asyncio.run(benchmark_serving(args))
    # except Exception as e:
    #     print("Error: failed to run benchmark, with exception:", e)
    #     BenchmarkMetrics().write_to_file(args)
    
if __name__ == "__main__":
    main()