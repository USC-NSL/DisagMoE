from disagmoe.frontend.controller import init_controller, Controller, AsyncResult
from disagmoe.utils.placement import ModelPlacement, ClusterConfig, get_model_placement
from disagmoe.utils.utils import StepInfo
from disagmoe.utils.metrics import Metric
from disagmoe.utils.constants import *
from disagmoe.config import (
    ModelConfig,
    CacheConfig,
    mixtral_config,
    qwen3_235b_config,
    qwen3_30b_config,
    gptoss_120b_config,
    glm45air_106b_config,
    glm45air_half_config,
    EngineConfig,
)
from disagmoe.frontend.datatypes import SloStat, TraceContext, SamplerStepInfo
from workload import PoissonGenerator, Workload, UniformGenerator, DatasetGenerator, get_generator
from utils import get_parser_base
import disagmoe_c as c
from disagmoe.utils.logger import new_logger
from typing import List, Dict, Tuple, Optional, Sequence, Union
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

master: Controller = None
logger = new_logger("Serving")

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
    
    peak_itl_median_ms: float = -1
    peak_itl_p99_ms: float = -1
    
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
                f"itl_latency_p99: {self.itl_latency_p99_ms:.0f}ms\n" \
                f"peak_itl_median: {self.peak_itl_median_ms:.0f}ms\n" \
                f"peak_itl_p99: {self.peak_itl_p99_ms:.0f}ms\n"

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

def override_model_config_with_args(args, model_config: ModelConfig):
    model_config.num_layers = args.num_layers if args.num_layers is not None else model_config.num_layers
    model_config.num_experts = args.num_experts if args.num_experts is not None else model_config.num_experts
    model_config.top_k = args.topk if args.topk is not None else model_config.top_k
    model_config.num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else model_config.num_kv_heads
    return model_config

# For heterogeneous cluster
def resolve_expert_allocation(
    expert_allocation_path: Optional[str],
    worker_identities: Sequence[Dict[str, Union[str, int]]],
    total_num_experts: int,
) -> Tuple[Optional[List[int]], Optional[Dict[int, float]], Optional[Dict[int, dict]]]:
    if expert_allocation_path is None:
        return None, None, None

    with open(expert_allocation_path, "r") as f:
        config = json.load(f)

    assert isinstance(config, dict), "Expert allocation config must be a JSON object"
    allocations = config.get("allocations")
    assert isinstance(allocations, list), "Expert allocation config must contain a list field 'allocations'"

    num_devices = len(worker_identities)
    expert_allocation = [0 for _ in range(num_devices)]
    identity_keys = ["host_ip", "ray_node_id", "ray_actor_id", "cuda_device"]

    for idx, entry in enumerate(allocations):
        assert isinstance(entry, dict), f"Allocation entry {idx} must be a JSON object"
        assert "num_experts" in entry, f"Allocation entry {idx} must contain 'num_experts'"
        assert "cuda_device" in entry, f"Allocation entry {idx} must contain 'cuda_device'"
        assert ("host_ip" in entry) or ("ray_node_id" in entry) or ("ray_actor_id" in entry), (
            f"Allocation entry {idx} must include one of 'host_ip', 'ray_node_id', or 'ray_actor_id'"
        )
        num_experts = entry["num_experts"]
        assert type(num_experts) is int and num_experts >= 0, f"Allocation entry {idx} has invalid 'num_experts': {num_experts}"

        selectors = {k: str(entry[k]) for k in identity_keys if k in entry}
        assert len(selectors) > 0, (
            f"Allocation entry {idx} must include at least one identity field among {identity_keys}"
        )

        matched = []
        for worker in worker_identities:
            ok = True
            for k, v in selectors.items():
                if str(worker.get(k, "")) != v:
                    ok = False
                    break
            if ok:
                matched.append(worker)

        assert len(matched) == 1, (
            f"Allocation entry {idx} matched {len(matched)} workers, expected exactly one. "
            f"selectors={selectors}"
        )

        device_id = matched[0].get("device_id")
        assert type(device_id) is int and 0 <= device_id < num_devices, (
            f"Allocation entry {idx} resolved invalid device_id: {device_id}"
        )
        expert_allocation[device_id] += num_experts

    assert sum(expert_allocation) == total_num_experts, (
        f"Total allocated experts ({sum(expert_allocation)}) must match model config ({total_num_experts})"
    )

    attn_dp_weights = None
    raw_weights = config.get("attn_dp_weights")
    if raw_weights is not None:
        attn_dp_weights = {int(k): float(v) for k, v in raw_weights.items()}

    per_device_config = None
    raw_pdc = config.get("per_device_config")
    if raw_pdc is not None:
        per_device_config = {int(k): v for k, v in raw_pdc.items()}

    return expert_allocation, attn_dp_weights, per_device_config

def launch(args):
    # Select transport in C++ backend (default from CLI is zmq)
    c.select_transport(args.transport)
    cluster_config = ClusterConfig(n_node=args.num_nodes, n_gpu=args.num_gpus)

    if args.model == "qwen3_235b":
        model_config = qwen3_235b_config
    elif args.model == "qwen3_30b":
        model_config = qwen3_30b_config
    elif args.model == "gptoss_120b":
        model_config = gptoss_120b_config
    elif args.model == "glm45air_106b":
        model_config = glm45air_106b_config
    elif args.model == "glm45air_half":
        model_config = glm45air_half_config
    elif args.model == "mixtral":
        model_config = mixtral_config
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model_config = override_model_config_with_args(args, model_config)
    model_config.ep_size = args.ep_size
    model_config.tp_size = args.tp_size
    model_config.dp_size = args.dp_size
    model_config.enable_trace = args.trace
    model_config.attn_qkv_quant = None if args.attn_qkv_quant in (None, "", "none") else args.attn_qkv_quant
    model_config.moe_linear_quant = None if getattr(args, "moe_linear_quant", None) in (None, "", "none") else args.moe_linear_quant
    model_config.num_shared_experts = getattr(args, "num_shared_experts", 0) or 0
    model_config.shared_expert_intermediate_size = getattr(args, "shared_expert_intermediate_size", None)
    
    engine_config = EngineConfig(
        enable_cuda_graph_attn=args.cuda_graph_attn,
        enable_cuda_graph_expert=args.cuda_graph_expert,
        enable_grouped_gemm=not args.serial_gemm and not args.expert_wise_schedule,
        less_than_sm90=getattr(args, "less_than_sm90", False),
        max_batch_size_attn=args.max_batch_size_attn,
        max_attn_graph_bsz=args.max_attn_graph_bsz,
        max_batch_size_expert=args.max_batch_size_expert,
        max_pending_sends=args.max_pending_sends,
        unified_scheduler_type=getattr(args, "unified_scheduler_type", "flfs"),
        defrag_weight_decay=getattr(args, "defrag_weight_decay"),
        defrag_lookahead_steps=getattr(args, "defrag_lookahead_steps"),
        defrag_lookback_steps=getattr(args, "defrag_lookback_steps"),
        enable_advanced_logging=getattr(args, "enable_advanced_logging", False),
        advanced_logging_dir=getattr(args, "advanced_logging_dir", "./advanced_logs"),
        advanced_logging_sample_rate=getattr(args, "advanced_logging_sample_rate", 0.1),
    )

    global master

    master = init_controller(
        cluster_config.n_node, 
        cluster_config.n_gpu, 
        host_ifname=getattr(args, "host_ifname", ""),
        nccl_ib_hca=getattr(args, "nccl_ib_hca", ""),
        nccl_ib_gid_index=getattr(args, "nccl_ib_gid_index", ""),
        expert_wise_schedule=args.expert_wise_schedule,
        enable_nsys=args.nsys
    )

    # For heterogeneous cluster
    worker_identities = master.get_worker_identities()
    expert_allocation, attn_dp_weights, per_device_config = resolve_expert_allocation(
        getattr(args, "expert_allocation_path", None),
        worker_identities,
        model_config.num_experts,
    )

    mp = get_model_placement(model_config, cluster_config, args.placement, 
                             step_attn=args.step_attn, step_expert=args.step_expert, 
                             zigzag_attn=args.zigzag_attn,
                             expert_allocation=expert_allocation)
    # mp = get_model_placement(model_config, cluster_config, "interleave")

    cache_config = CacheConfig(args.block_size, args.gpu_usage, 2, "auto")

    master.init_engine(args.transport, mp, model_config, engine_config, cache_config,
                      attn_dp_weights=attn_dp_weights,
                      per_device_config=per_device_config,
                      gate_profile_file=args.gate_profile_file,
                      dp_policy=getattr(args, "dp_policy", None))
    
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
        
    from plotter.namer import get_trace_name, get_queue_length_name, get_trace_metrics_name
    
    trace_dir = os.path.dirname(args.file)

    with gzip.open(get_trace_name(args, trace_dir), "w") as f:
        f.write(json.dumps(events).encode("utf-8"))

    with open(get_trace_metrics_name(args, trace_dir), "w") as f:
        json.dump(metrics, f)
        
    with open(get_queue_length_name(args, trace_dir), "wb") as f:
        pickle.dump(queue_length_per_step, f)

def analyze_throughput(args, 
                       req_submit_timestamps: List[float],
                       req_finish_timestamps: List[float],
                       sampler_step_infos: List[SamplerStepInfo], 
                       attn_queueing_delays: List[List[float]],
                       exp_queueing_delays: List[List[float]],
                       t_submitted: Dict[int, int],
                       slo_stats: List[SloStat]):
    from plotter.namer import get_sampler_step_name, get_worker_queueing_delay_name, \
                                        get_ttft_name, get_req_finish_time_name, get_req_submit_time_name
    trace_dir = os.path.dirname(args.file)
    # request submit timestamp
    req_submit_fn = get_req_submit_time_name(args, trace_dir)
    req_submit_df = pd.DataFrame(req_submit_timestamps)
    req_submit_df.to_csv(req_submit_fn, index=False)
    
    # request finish timestamp
    req_finish_fn = get_req_finish_time_name(args, trace_dir)
    req_finish_df = pd.DataFrame(req_finish_timestamps)
    req_finish_df.to_csv(req_finish_fn, index=False)
    
    # sampler throughput
    sampler_fn = get_sampler_step_name(args, trace_dir)
    sampler_df = pd.DataFrame([asdict(info) for info in sampler_step_infos])
    sampler_df.to_csv(sampler_fn, index=False)
    
    def get_peak_throughput(time_bin=5):
        sampler_df['time_stamp'] = (sampler_df['time_stamp'] - sampler_df['time_stamp'].iloc[0]) / (10 ** 3)
        seg = int((sampler_df['time_stamp'].iloc[-1] - sampler_df['time_stamp'].iloc[0] + time_bin - 1) // time_bin)
        time_bins = [sampler_df['time_stamp'].iloc[0] + i * time_bin for i in range(seg + 1)]
        time_sums = sampler_df.groupby(pd.cut(sampler_df['time_stamp'], bins=time_bins))['num_tokens'].sum()
        time_sums /= time_bin
        
        window = getattr(args, 'analyze_throughput_window', None)
        if window:
            win_lo, win_hi = [float(x) for x in window.split(',')]
            t0 = sampler_df['time_stamp'].iloc[0]
            mask = [(t0 + win_lo <= (iv.left + iv.right) / 2 <= t0 + win_hi) for iv in time_sums.index]
            peak_throughput_range = time_sums[mask]
        else:
            num_bins = len(time_sums)
            peak_throughput_time_range = 60
            step = peak_throughput_time_range // 2 // time_bin
            peak_throughput_range = time_sums[num_bins // 2 - step : num_bins // 2 + step]
        if len(peak_throughput_range) == 0:
            return sum(time_sums) / len(time_sums)
        return sum(peak_throughput_range) / len(peak_throughput_range)
    
    # queueing delay
    attn_fn = get_worker_queueing_delay_name(args, "attn", trace_dir)
    attn_df = pd.DataFrame(attn_queueing_delays)
    attn_df.to_csv(attn_fn, index=False)
    
    exp_fn = get_worker_queueing_delay_name(args, "exp", trace_dir)
    exp_df = pd.DataFrame(exp_queueing_delays)
    exp_df.to_csv(exp_fn, index=False)
    
    # TTFT
    ttft_fn = get_ttft_name(args, trace_dir)
    ttft_df = pd.DataFrame([
        stat.t_prefill_std - t_submitted[stat.req_id]
            for stat in slo_stats
    ])
    ttft_df.to_csv(ttft_fn, index=False)
    
    def get_peak_itl(peak_time_range=60):
        all_itls = []
        for stat in slo_stats:
            if stat.t_token_timestamps is None or len(stat.t_token_timestamps) < 2:
                continue
            for i in range(1, len(stat.t_token_timestamps)):
                t = stat.t_token_timestamps[i]
                itl_ms = (stat.t_token_timestamps[i] - stat.t_token_timestamps[i - 1]) * 1e3
                all_itls.append((t, itl_ms))
        
        if not all_itls:
            return -1.0, -1.0
        
        all_itls.sort(key=lambda x: x[0])
        t_start = all_itls[0][0]
        t_end = all_itls[-1][0]
        
        window = getattr(args, 'analyze_throughput_window', None)
        if window:
            win_lo, win_hi = [float(x) for x in window.split(',')]
            t_lo = t_start + win_lo
            t_hi = t_start + win_hi
        else:
            t_mid = (t_start + t_end) / 2
            t_lo = t_mid - peak_time_range / 2
            t_hi = t_mid + peak_time_range / 2
        
        peak_itls = np.array([itl for t, itl in all_itls if t_lo <= t <= t_hi])
        if len(peak_itls) == 0:
            return -1.0, -1.0
        
        return float(np.median(peak_itls)), float(np.percentile(peak_itls, 99))
    
    peak_throughput = get_peak_throughput()
    peak_itl_median, peak_itl_p99 = get_peak_itl()
    return peak_throughput, peak_itl_median, peak_itl_p99

    
async def run_benchmark(master: Controller, args, 
                        generator_type, num_requests, 
                        min_input_len, max_input_len, 
                        min_output_len, max_output_len, 
                        rate, warmup=False):
    GeneratorType = get_generator(generator_type)
    if GeneratorType is DatasetGenerator:
        dataset_path = getattr(args, "dataset_path", None)
        assert dataset_path is not None, "--dataset-path required when using --generator-type=dataset"
        max_context_len = getattr(args, "dataset_max_context_len", None)
        generator = DatasetGenerator(rate, 1, dataset_path, max_context_len)
    else:
        generator = GeneratorType(rate, 1, min_input_len, max_input_len, min_output_len, max_output_len)
    workload = generator.generate_num(num_requests)
    pbar = tqdm.tqdm(total=num_requests)
    t_start = time.perf_counter()
    tasks = []
    req_submit_timestamps = []
    req_finish_timestamps = []
    logger.info(f"generating requests at rate {args.rate} s/req, in total {args.num_requests} requests")
    for i in range(num_requests):
        t_elapsed = time.perf_counter() - t_start
        arrival, input_len, output_len = workload[i]
        if t_elapsed < arrival:
            await asyncio.sleep(arrival - t_elapsed)
        req_submit_timestamps.append(time.perf_counter() - t_start)
        resp = master.put_single_request(input_len, output_len)
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
    warmup_input = 100
    warmup_output = 50
    await run_benchmark(
        master, args, args.generator_type, _num_warmup_requests, 
        warmup_input, warmup_input + 1, warmup_output, warmup_output + 1,
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
        throughput, peak_itl_median, peak_itl_p99 = analyze_throughput(args, 
                            req_submit_timestamps,
                            req_finish_timestamps,
                            sampler_step_infos, 
                            attn_delays, exp_delays,
                            t_submitted, results)
        metrics.token_throughput = throughput
        metrics.peak_itl_median_ms = peak_itl_median
        metrics.peak_itl_p99_ms = peak_itl_p99

    if getattr(args, "enable_advanced_logging", False):
        _adv_dir = getattr(args, "advanced_logging_dir", "./advanced_logs")
        master.dump_advanced_logs(output_dir=_adv_dir)
        logger.info(f"Advanced logs dumped to {_adv_dir}")
         
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
        master.init_profile(profile_dir=args.profile_dir)

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
    if getattr(args, "expert_allocation_path", None) is None:
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
