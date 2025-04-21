from disagmoe.frontend.controller import init_controller, Controller, AsyncResult
from disagmoe.utils.placement import ModelPlacement, ClusterConfig, get_model_placement
from disagmoe.utils.utils import StepInfo
from disagmoe.utils.metrics import Metric
from disagmoe.utils.constants import *
from disagmoe.config import ModelConfig, CacheConfig, mixtral_config, SamplingConfig
from disagmoe.frontend.datatypes import SloStat, TraceContext, SamplerStepInfo
from benchmark.workload import PoissonGenerator, Workload, UniformGenerator, get_generator
from benchmark.utils import get_parser_base
from disagmoe.utils.logger import get_logger
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
import pickle

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

master: Controller = None
logger = get_logger("Serving")


@dataclass
class BenchmarkMetrics:
    e2e_duration: float = -1
    req_throughput: float = -1
    token_throughput: float = -1
    
    req_latency_mean_ms: float = -1
    req_latency_median_ms: float = -1
    req_latency_p99_ms: float = -1
    
    itl_latency_mean_ms: float = -1
    itl_latency_median_ms: float = -1
    itl_latency_p99_ms: float = -1
    
    def __repr__(self):
        return f"Metrics: \n" \
                f"e2e_duration: {self.e2e_duration:.2f}s\n" \
                f"req_throughput: {self.req_throughput:.0f} req/s\n" \
                f"token_throughput: {self.token_throughput:.0f} tokens/s\n" \
                f"req_latency_mean: {self.req_latency_mean_ms:.0f}ms\n" \
                f"req_latency_median: {self.req_latency_median_ms:.0f}ms\n" \
                f"req_latency_p99: {self.req_latency_p99_ms:.0f}ms\n" \
                f"itl_latency_mean: {self.itl_latency_mean_ms:.0f}ms\n" \
                f"itl_latency_median: {self.itl_latency_median_ms:.0f}ms\n" \
                f"itl_latency_p99: {self.itl_latency_p99_ms:.0f}ms\n"

    def write_to_file(self, args):
        filename = args.file
        metrics = { k: int(v) for k, v in self.__dict__.items()}
        try:
            import pandas as pd
            if not os.path.exists(filename):
                df = pd.DataFrame(columns=["num_requests", "rate", "DP_size", "EP_size", 
                                           "num_experts"] + list(metrics.keys()))
            else:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filename)
                else:
                    df = pd.read_excel(filename)
            new_row = {
                "num_requests": args.num_requests,
                "rate": args.rate,
                "DP_size": args.dp_size,
                "EP_size": args.ep_size,
                "num_experts": args.num_experts,
                **metrics
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
    model_config.enable_grouped_gemm = not args.serial_gemm and not args.expert_wise_schedule
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

    master = init_controller(
        cluster_config.n_node, 
        cluster_config.n_gpu, 
        expert_wise_schedule=args.expert_wise_schedule,
        enable_nsys=args.nsys
    )

    cache_config = CacheConfig(args.block_size, args.gpu_usage, 2, "auto",
                               num_gpu_blocks=args.num_blocks + RESERVED_BLOCKS if args.num_blocks else None, # default should be None
                               num_reserved_blocks=RESERVED_BLOCKS)

    sampling_config = SamplingConfig(min_output_len=args.min_output_len, max_output_len=args.max_output_len)
    
    master.init_engine(mp, model_config, cache_config, sampling_config)
    
    master.start_engine()
    
    return master
    
   
async def process_response(resp: AsyncResult, req_finish_timestamps: List[float], pbar):
    slo_stat = await resp.get()
    # print(f">>> Response received: {resp.req_id}")
    req_finish_timestamps.append(time.perf_counter())
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
        
    metrics = BenchmarkMetrics(
        e2e_duration=duration,
        req_throughput=num_reqs / duration,
        token_throughput=total_output_tokens / duration,
        
        req_latency_mean_ms=np.mean(req_latency) * 1000,
        req_latency_median_ms=np.median(req_latency) * 1000,
        req_latency_p99_ms=np.percentile(req_latency, 99) * 1000,
        
        itl_latency_mean_ms=np.mean(itls) * 1000,
        itl_latency_median_ms=np.median(itls) * 1000,
        itl_latency_p99_ms=np.percentile(itls, 99) * 1000,
    )
    
    return metrics

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
                        step_stats: List[Tuple[List[StepInfo], Dict[int, List[TraceContext]], Metric]]):
    def ms_to_us(ms):
        return ms * 1000
    events = []
    metrics = {}
    
    queue_length_per_step = []
    
    for pid, (worker_stats, p_traces, metric) in enumerate(step_stats):
        layers = set()
        for step_info in worker_stats:
            if step_info.layer_id >= 0:
                layers.add(step_info.layer_id)
        layers = sorted(list(layers))
        
        worker_queue_length_per_step = {layer: [] for layer in layers}
        step_start_time_ms = []
        step_executed_layer = []
        for step_info in worker_stats:
            # empty step is labeled as -1
            if step_info.layer_id >= 0:
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
            step_start_time_ms.append(step_info.start_timestamp_ms)
            step_executed_layer.append(step_info.internal_layer_id)
            for layer in layers:
                if layer in step_info.pool_snapshot:
                    worker_queue_length_per_step[layer].append(step_info.pool_snapshot[layer])
                else:
                    worker_queue_length_per_step[layer].append(0)
        
        queue_length_per_step.append((worker_queue_length_per_step, step_executed_layer, step_start_time_ms))
        
        if args.enable_trace_detail:
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
        
    from benchmark.plotter.namer import get_trace_name, get_queue_length_name, get_trace_metrics_name

    with gzip.open(get_trace_name(args), "w") as f:
        f.write(json.dumps(events).encode("utf-8"))

    with open(get_trace_metrics_name(args), "w") as f:
        json.dump(metrics, f)
        
    with open(get_queue_length_name(args), "wb") as f:
        pickle.dump(queue_length_per_step, f)

def analyze_throughput(args, 
                       req_submit_timestamps: List[float],
                       req_finish_timestamps: List[float],
                       sampler_step_infos: List[SamplerStepInfo], 
                       attn_queueing_delays: List[List[float]],
                       exp_queueing_delays: List[List[float]],
                       t_submitted: Dict[int, int],
                       slo_stats: List[SloStat]):
    from benchmark.plotter.namer import get_sampler_step_name, get_worker_queueing_delay_name, \
                                        get_ttft_name, get_req_finish_time_name, get_req_submit_time_name
    
    # request submit timestamp
    req_submit_fn = get_req_submit_time_name(args)
    req_submit_df = pd.DataFrame(req_submit_timestamps)
    req_submit_df.to_csv(req_submit_fn, index=False)
    
    # request finish timestamp
    req_finish_fn = get_req_finish_time_name(args)
    req_finish_df = pd.DataFrame(req_finish_timestamps)
    req_finish_df.to_csv(req_finish_fn, index=False)
    
    # sampler throughput
    sampler_fn = get_sampler_step_name(args)
    sampler_df = pd.DataFrame([asdict(info) for info in sampler_step_infos])
    sampler_df.to_csv(sampler_fn, index=False)
    
    def get_peak_throughput(time_bin=5):
        sampler_df['time_stamp'] = (sampler_df['time_stamp'] - sampler_df['time_stamp'].iloc[0]) / (10 ** 3)
        seg = int((sampler_df['time_stamp'].iloc[-1] - sampler_df['time_stamp'].iloc[0] + time_bin - 1) // time_bin)
        time_bins = [sampler_df['time_stamp'].iloc[0] + i * time_bin for i in range(seg + 1)]
        time_sums = sampler_df.groupby(pd.cut(sampler_df['time_stamp'], bins=time_bins))['num_tokens'].sum()
        time_sums /= time_bin
        return time_sums.max()
    
    # queueing delay
    attn_fn = get_worker_queueing_delay_name(args, "attn")
    attn_df = pd.DataFrame(attn_queueing_delays)
    attn_df.to_csv(attn_fn, index=False)
    
    exp_fn = get_worker_queueing_delay_name(args, "exp")
    exp_df = pd.DataFrame(exp_queueing_delays)
    exp_df.to_csv(exp_fn, index=False)
    
    # TTFT
    ttft_fn = get_ttft_name(args)
    ttft_df = pd.DataFrame([
        stat.t_prefill_std - t_submitted[stat.req_id]
            for stat in slo_stats
    ])
    ttft_df.to_csv(ttft_fn, index=False)
    
    return get_peak_throughput()

    
async def run_benchmark(master: Controller, args, 
                        generator_type, num_requests, 
                        min_input_len, max_input_len, 
                        min_output_len, max_output_len, 
                        rate, warmup=False):
    GeneratorType = get_generator(generator_type)
    generator = GeneratorType(rate, 1, min_input_len, max_input_len)
    workload = generator.generate_num(num_requests)
    pbar = tqdm.tqdm(total=num_requests)
    t_start = time.perf_counter()
    tasks = []
    req_submit_timestamps = []
    req_finish_timestamps = []
    logger.info(f"generating requests at rate {args.rate} s/req, in total {args.num_requests} requests")
    for i in range(num_requests):
        t_elapsed = time.perf_counter() - t_start
        arrival, input_len = workload[i]
        if t_elapsed < arrival:
            await asyncio.sleep(arrival - t_elapsed)
        req_submit_timestamps.append(time.perf_counter() - t_start)
        resp = master.put_single_request(input_len)
        tasks.append(asyncio.create_task(process_response(resp, req_finish_timestamps, pbar)))
    
    results: List[SloStat] = await asyncio.gather(*tasks)
    t_duration = time.perf_counter() - t_start
    pbar.close()
    
    if warmup:
        return None
    
    logger.info("Benchmark finished, now analyznig results ...")

    req_finish_timestamps.sort()
    for i in range(num_requests):
        req_finish_timestamps[i] -= t_start

    return results, req_submit_timestamps, req_finish_timestamps, t_duration

async def benchmark_warmup(master: Controller, args):
    logger.info("Now running warmup ...")
    _num_warmup_requests = 10
    _num_warmup_rate = 5
    await run_benchmark(
        master, args, args.generator_type, _num_warmup_requests, 
        args.min_input_len, args.max_input_len,
        args.min_output_len, args.max_output_len, 
        _num_warmup_rate, warmup=True
    )
    master.reset()
    logger.info("Warmup done.")

def post_benchmark(master, args, results, req_submit_timestamps, req_finish_timestamps, duration):
    metrics = analyze_results(results, duration)
    
    if args.trace:
        step_stats = master.fetch_step_stats()
        generate_step_trace(args, step_stats)
    
    if args.analyze_throughput:
        sampler_step_infos = master.fetch_sampler_step_infos()
        attn_delays, exp_delays = master.fetch_queueing_delays()
        t_submitted = master.fetch_submitted_time()
        throughput = analyze_throughput(args, 
                            req_submit_timestamps,
                            req_finish_timestamps,
                            sampler_step_infos, 
                            attn_delays, exp_delays,
                            t_submitted, results)
        metrics.token_throughput = throughput
        
    metrics.write_to_file(args)
    logger.info("Results written to file.")
    return metrics

async def benchmark_serving(
    master: Controller,
    args, 
    is_api_server: bool = False
):
    assert master is not None, "master is not initialized"
    
    if not is_api_server:
        master.start_polling_results()
        await master.start_scheduler()
        await benchmark_warmup(master, args)
        
    if args.profile_dir is not None:
        master.start_profile(args.profile_dir)

    # run benchmark
    logger.info("Now running benchmark.")
    results, req_submit_timestamps, req_finish_timestamps, duration = await run_benchmark(
        master, args, args.generator_type, args.num_requests,
        args.min_input_len, args.max_input_len,
        args.min_output_len, args.max_output_len,  args.rate
    )
    
    if not is_api_server:
        await master.stop_scheduler()
        master.stop_workers()
        
    master.stop_profile()
    
    metrics = post_benchmark(master, args, results, req_submit_timestamps, req_finish_timestamps, duration)
    
    master.reset()
    
    return metrics
    
def get_args():
    args = get_parser_base().parse_args()
    
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
    
    try:
        launch(args)
        metrics = asyncio.run(benchmark_serving(master, args))
        print(metrics)  # Or do something else with the metrics
    except Exception as e:
        print("Error: failed to run benchmark, with exception:", e.with_traceback(None))
        BenchmarkMetrics().write_to_file(args)
    
if __name__ == "__main__":
    main()