from disagmoe.frontend.controller import init_controller, Controller, AsyncResult
from disagmoe.utils.placement import ModelPlacement, ClusterConfig, get_model_placement
from disagmoe.utils.constants import *
from disagmoe.config import ModelConfig, CacheConfig, duo_expert_mixtral, SamplingConfig
from disagmoe.frontend.datatypes import SloStat
from typing import List
from argparse import ArgumentParser

import asyncio
import time
import tqdm
from dataclasses import dataclass
import numpy as np

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
    cluster_config = ClusterConfig(n_node=1, n_gpu=3, 
                                id_tokenizer=tokenizer, 
                                id_sampler=sampler)

    model_config = duo_expert_mixtral
    model_config.num_layers = 32
    model_config.ep_size = 2
    model_config.num_experts = 8
    model_config.tp_size = 1
    model_config.tp_enable_inter_group = False

    mp = get_model_placement(model_config, cluster_config, "interleave")

    global master

    master = init_controller(cluster_config.n_node, cluster_config.n_gpu)

    cache_config = CacheConfig(BLOCK_SIZE, 0.8, 2, "auto", 
                                num_gpu_blocks=NUM_BLOCKS, 
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
    
    master.stop_workers()
    
    
def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("-r", "--rate", type=float, default=0, help="rate of incoming requests, seconds per request")
    parser.add_argument("-i", "--input-len", type=int, default=1, help="length of input sequence")
    parser.add_argument("-o", "--output-len", type=int, default=32, help="length of output sequence")
    parser.add_argument("-n", "--num-requests", type=int, default=1000, help="number of requests to generate")
    parser.add_argument("-p", "--profile-dir", type=str, default=None, help="directory to store torch profiler output")
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    launch(args)
    
    asyncio.run(benchmark_serving(args))
    
if __name__ == "__main__":
    main()