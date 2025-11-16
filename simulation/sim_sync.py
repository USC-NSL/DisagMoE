from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import torch

from util import (
    expert_compute_time_lookup_table_from_profile,
    build_profile_router,
)
from disagmoe.models.gate import ProfileDrivenRouter


# ------------------------------
# Configurable parameters
# ------------------------------

EP_GROUP_SIZE      = 16      # "n": number of expert workers globally
TOTAL_EXPERT_COUNT = 128     # total experts per layer
MAX_BATCH_SIZE     = 512
NUM_LAYERS         = 48      # number of expert layers
TOTAL_REQUESTS     = 1024
TOKENS_PER_REQUEST = 512
TOTAL_TOKENS       = TOTAL_REQUESTS * TOKENS_PER_REQUEST
GLOBAL_REQUEST_MAX_BATCH_SIZE = EP_GROUP_SIZE * 256  # ??????? max concurrent active requests

ARRIVAL_RATE       = 50.0   # lambda for Poisson arrivals (requests / tick)
# try for each attention worker, devide the arrival rate by the number of attention workers

ATTN_SERVICE_T     = 2    # time (ticks) per token at attention worker
TICKS_PER_MILLISECOND = 10  # 0.1 ms per tick

NET_T_EXPERT_TO_ATTN  = 0.1  # fixed network delay expert -> attention in #ticks, 10us
NET_T_ATTN_TO_EXPERT  = 0.1  # fixed network delay attention -> expert in #ticks, 10us

GROUP_GEMM_SPEEDUP_FACTOR = 0.7  # speedup factor for grouped GEMM vs single-expert profile

RNG_SEED = 42

EXPERT_COMPUTE_PROFILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "expert_costs_profiles",
    "Qwen3-30B.csv",
)

ROUTING_TOP_K = 8
PROFILE_ROUTING_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "gating_profiles",
        "gating_sharegptv3_155.parquet",
    )
)


# ------------------------------
# Token representation
# ------------------------------


@dataclass
class Token:
    tid: int
    birth_time: float
    request_id: int
    token_index: int
    layer_fanout: Dict[int, int] = field(default_factory=dict)


@dataclass
class RequestState:
    request_id: int
    arrival_time: float
    next_token_index: int = 0
    completed: bool = False


class ProgressTracker:
    """Simple stdout progress bar with ETA estimation."""

    def __init__(self, total_tokens: int, bar_width: int = 30, min_interval: float = 0.5):
        self.total = max(0, int(total_tokens))
        self.bar_width = bar_width
        self.min_interval = min_interval
        self.start_time = time.perf_counter()
        self.last_render = 0.0
        self.finished = False

    def update(self, completed: int, force: bool = False):
        if self.finished and not force:
            return

        now = time.perf_counter()
        if (
            not force
            and completed < self.total
            and (now - self.last_render) < self.min_interval
        ):
            return

        pct = 1.0 if self.total == 0 else min(1.0, max(0.0, completed / float(self.total)))
        elapsed = now - self.start_time
        eta = self._estimate_eta(elapsed, pct, completed)

        bar_fill = int(self.bar_width * pct)
        bar = "#" * bar_fill + "-" * (self.bar_width - bar_fill)
        eta_str = self._format_duration(eta)
        elapsed_str = self._format_duration(elapsed)

        sys.stdout.write(
            f"\r[{bar}] {pct * 100:5.1f}% ({completed}/{self.total}) "
            f"Elapsed {elapsed_str} ETA {eta_str}"
        )
        sys.stdout.flush()
        self.last_render = now

        if completed >= self.total and not self.finished:
            self.finished = True
            sys.stdout.write("\n")
            sys.stdout.flush()

    def finalize(self):
        if not self.finished:
            self.update(self.total, force=True)
            if not self.finished:
                self.finished = True
                sys.stdout.write("\n")
                sys.stdout.flush()

    def _estimate_eta(self, elapsed: float, pct: float, completed: int) -> float:
        if completed >= self.total or pct <= 0.0:
            return 0.0
        remaining = (1.0 - pct) * elapsed / pct
        return remaining

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds is None or not (seconds < float("inf")):
            return "--:--"
        if seconds <= 0:
            return "00:00"
        total_seconds = int(seconds + 0.5)
        mins, sec = divmod(total_seconds, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 99:
            return ">99h"
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{sec:02d}"
        return f"{mins:02d}:{sec:02d}"


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0.0:
        return min(values)
    if pct >= 1.0:
        return max(values)
    sorted_vals = sorted(values)
    k = pct * (len(sorted_vals) - 1)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_vals[int(k)]
    frac = k - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * frac


def generate_request_arrivals(total_requests: int, arrival_rate: float, rng: random.Random) -> List[float]:
    """
    Generate absolute arrival times for each request following a Poisson process.
    The first request arrives at time 0, matching the async simulator's behavior.
    """
    if total_requests <= 0:
        return []
    arrivals: List[float] = [0.0]
    current = 0.0
    for _ in range(1, total_requests):
        if arrival_rate > 0.0:
            current += rng.expovariate(arrival_rate)
        arrivals.append(current)
    return arrivals


class SyncMoESimulator:
    """
    Centralized synchronous MoE simulator that advances tokens layer-by-layer.
    Each iteration processes up to GLOBAL_REQUEST_MAX_BATCH_SIZE active requests,
    routing their current token through every layer before moving to the next
    autoregressive position.
    """

    def __init__(
        self,
        request_arrivals: List[float],
        tokens_per_request: int,
        num_layers: int,
        total_expert_count: int,
        ep_group_size: int,
        max_batch_size: int,
        attn_service_t: float,
        net_t_attn_to_expert: float,
        net_t_expert_to_attn: float,
        compute_time_lookup: Dict[int, float],
        profile_router: ProfileDrivenRouter,
        progress_tracker: ProgressTracker,
        global_request_max_batch_size: int,
    ):
        if total_expert_count % ep_group_size != 0:
            raise ValueError("total_expert_count must be divisible by ep_group_size")

        self.tokens_per_request = tokens_per_request
        self.num_layers = num_layers
        self.total_expert_count = total_expert_count
        self.ep_group_size = ep_group_size
        self.max_batch_size = max_batch_size
        self.attn_service_t = attn_service_t
        self.net_t_attn_to_expert = net_t_attn_to_expert
        self.net_t_expert_to_attn = net_t_expert_to_attn
        self.compute_time_lookup = compute_time_lookup
        self.profile_router = profile_router
        self.progress_tracker = progress_tracker
        self.global_request_max_batch_size = max(1, int(global_request_max_batch_size))

        self.total_requests = len(request_arrivals)
        self.expected_total_tokens = self.total_requests * self.tokens_per_request

        self.requests: List[RequestState] = [
            RequestState(request_id=rid, arrival_time=request_arrivals[rid])
            for rid in range(self.total_requests)
        ]
        self.request_lookup: Dict[int, RequestState] = {
            req.request_id: req for req in self.requests
        }

        self.current_time = 0.0
        self.arrival_index = 0
        self.waiting_queue: Deque[RequestState] = deque()
        self.active_requests: List[RequestState] = []

        self.next_tid = 0
        self.completion_times: Dict[int, float] = {}
        self.token_latencies: Dict[int, float] = {}
        self.request_latency_values: List[float] = []

        self.completed_tokens = 0

        self.queues_per_worker = self.total_expert_count // self.ep_group_size
        self._router_device = torch.device("cpu")
        self._router_dtype = torch.float32

        # Per-layer metrics for imbalance analysis
        self.layer_expert_runtimes: List[float] = []
        self.layer_wait_imbalances: List[float] = []

    def run(self):
        while self.completed_tokens < self.expected_total_tokens:
            self._release_new_arrivals()
            self._fill_active_requests()

            if not self.active_requests:
                if not self._advance_to_next_arrival():
                    break
                continue

            tokens = self._spawn_tokens_for_iteration()
            if not tokens:
                # Should not happen, but guard against zero-token iterations.
                if not self._advance_to_next_arrival():
                    break
                continue

            iteration_duration = self._run_one_iteration(tokens)
            self.current_time += iteration_duration
            self._finalize_iteration(tokens)

        self.progress_tracker.finalize()

        # Aggregate per-layer metrics across all layers and iterations.
        if self.layer_expert_runtimes:
            avg_layer_expert_runtime = (
                sum(self.layer_expert_runtimes) / len(self.layer_expert_runtimes)
            )
        else:
            avg_layer_expert_runtime = 0.0

        if self.layer_wait_imbalances:
            avg_layer_wait_imbalance = (
                sum(self.layer_wait_imbalances) / len(self.layer_wait_imbalances)
            )
        else:
            avg_layer_wait_imbalance = 0.0

        return {
            "completion_times": self.completion_times,
            "token_latencies": self.token_latencies,
            "request_latency_values": self.request_latency_values,
            "makespan": self.current_time,
            "avg_layer_expert_runtime": avg_layer_expert_runtime,
            "avg_layer_wait_imbalance": avg_layer_wait_imbalance,
        }

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _release_new_arrivals(self):
        while (
            self.arrival_index < self.total_requests
            and self.requests[self.arrival_index].arrival_time <= self.current_time
        ):
            self.waiting_queue.append(self.requests[self.arrival_index])
            self.arrival_index += 1

    def _advance_to_next_arrival(self) -> bool:
        if self.arrival_index >= self.total_requests:
            return False
        next_time = self.requests[self.arrival_index].arrival_time
        if next_time > self.current_time:
            self.current_time = next_time
        self._release_new_arrivals()
        return True

    def _fill_active_requests(self):
        while (
            len(self.active_requests) < self.global_request_max_batch_size
            and self.waiting_queue
        ):
            candidate = self.waiting_queue.popleft()
            if candidate.completed:
                continue
            self.active_requests.append(candidate)

    def _spawn_tokens_for_iteration(self) -> List[Token]:
        tokens: List[Token] = []
        for request in self.active_requests:
            if request.completed or request.next_token_index >= self.tokens_per_request:
                continue
            token = Token(
                tid=self.next_tid,
                birth_time=self.current_time,
                request_id=request.request_id,
                token_index=request.next_token_index,
            )
            self.next_tid += 1
            tokens.append(token)
        return tokens

    def _run_one_iteration(self, tokens: List[Token]) -> float:
        iteration_time = 0.0
        for layer_idx in range(self.num_layers):
            iteration_time += self._run_layer(layer_idx, tokens)
        return iteration_time

    def _run_layer(self, layer_idx: int, tokens: List[Token]) -> float:
        token_count = len(tokens)
        if token_count == 0:
            return 0.0

        attention_time = token_count * self.attn_service_t
        dispatch_time = token_count * self.net_t_attn_to_expert

        worker_loads = self._route_layer(layer_idx, tokens)
        worker_times = [
            self._simulate_worker_time(loads) for loads in worker_loads
        ]
        expert_time = max(worker_times) if worker_times else 0.0

        # Per-layer metrics:
        # - total layer runtime (from attention start to last expert completion)
        # - longest wait (from first worker completion to last worker completion)
        if worker_times:
            earliest_finish = min(worker_times)
            latest_finish = max(worker_times)
            longest_wait = max(0.0, latest_finish - earliest_finish)
        else:
            longest_wait = 0.0

        layer_runtime = attention_time + dispatch_time + expert_time
        self.layer_expert_runtimes.append(layer_runtime)
        self.layer_wait_imbalances.append(longest_wait)

        return layer_runtime

    def _route_layer(self, layer_idx: int, tokens: List[Token]) -> List[List[int]]:
        worker_queues: List[List[int]] = [
            [0 for _ in range(self.queues_per_worker)]
            for _ in range(self.ep_group_size)
        ]
        request_ids = [token.request_id for token in tokens]
        token_indices = torch.tensor(
            [token.token_index for token in tokens],
            device=self._router_device,
            dtype=torch.int64,
        )
        _, topk_ids = self.profile_router.route(
            request_ids=request_ids,
            token_indices=token_indices,
            layer_id=layer_idx,
            top_k=self.profile_router.top_k,
            device=self._router_device,
            dtype=self._router_dtype,
        )
        topk_ids_cpu = topk_ids.to("cpu").tolist()

        for token, expert_ids in zip(tokens, topk_ids_cpu):
            selected = [eid for eid in expert_ids if eid >= 0]
            if not selected:
                raise RuntimeError(
                    f"Router returned no experts for token {token.tid} at layer {layer_idx}"
                )
            token.layer_fanout[layer_idx] = len(selected)

            for expert_id in selected:
                if expert_id >= self.total_expert_count:
                    raise RuntimeError(
                        f"Expert id {expert_id} exceeds total experts {self.total_expert_count}"
                    )
                worker_idx = expert_id // self.queues_per_worker
                local_queue_idx = expert_id % self.queues_per_worker
                worker_queues[worker_idx][local_queue_idx] += 1
        return worker_queues

    def _simulate_worker_time(self, queue_lengths: List[int]) -> float:
        total_tokens = sum(queue_lengths)
        if total_tokens == 0:
            return 0.0

        queues = list(queue_lengths)
        elapsed = 0.0
        while total_tokens > 0:
            queue_idx = max(range(len(queues)), key=lambda i: queues[i])
            available = queues[queue_idx]
            if available <= 0:
                break
            batch = min(available, self.max_batch_size)
            queues[queue_idx] -= batch
            total_tokens -= batch
            compute_t = self.compute_time_lookup[batch]
            elapsed += compute_t + self.net_t_expert_to_attn
        return elapsed

    def _finalize_iteration(self, tokens: List[Token]):
        completion_time = self.current_time
        for token in tokens:
            self.completion_times[token.tid] = completion_time
            latency = completion_time - token.birth_time
            self.token_latencies[token.tid] = latency
            self.completed_tokens += 1

            req = self.request_lookup[token.request_id]
            req.next_token_index += 1
            if req.next_token_index >= self.tokens_per_request:
                req.completed = True
                self.request_latency_values.append(
                    completion_time - req.arrival_time
                )

        self.progress_tracker.update(self.completed_tokens)
        # Remove finished requests before the next iteration starts.
        self.active_requests = [req for req in self.active_requests if not req.completed]


# ------------------------------
# Main simulation driver
# ------------------------------


def run_simulation(
    ep_group_size=EP_GROUP_SIZE,
    total_expert_count=TOTAL_EXPERT_COUNT,
    max_batch_size=MAX_BATCH_SIZE,
    num_layers=NUM_LAYERS,
    total_requests=TOTAL_REQUESTS,
    tokens_per_request=TOKENS_PER_REQUEST,
    arrival_rate=ARRIVAL_RATE,
    attn_service_t=ATTN_SERVICE_T,
    net_t_attn_to_expert=NET_T_ATTN_TO_EXPERT,
    net_t_expert_to_attn=NET_T_EXPERT_TO_ATTN,
    ticks_per_millisecond=TICKS_PER_MILLISECOND,
    expert_profile_path=EXPERT_COMPUTE_PROFILE_PATH,
    profile_routing_path=PROFILE_ROUTING_PATH,
    routing_top_k=ROUTING_TOP_K,
    rng_seed=RNG_SEED,
    global_request_max_batch_size=GLOBAL_REQUEST_MAX_BATCH_SIZE,
):
    rng = random.Random(rng_seed)

    total_tokens_expected = total_requests * tokens_per_request
    progress_tracker = ProgressTracker(total_tokens_expected)

    base_compute_time_lookup = expert_compute_time_lookup_table_from_profile(
        expert_profile_path,
        max_batch_size,
        ticks_per_millisecond=ticks_per_millisecond,
    )

    # Apply group GEMM speedup factor for sync simulator.
    compute_time_lookup: Dict[int, float] = {
        bs: t * GROUP_GEMM_SPEEDUP_FACTOR for bs, t in base_compute_time_lookup.items()
    }

    profile_router = build_profile_router(
        profile_path=profile_routing_path,
        num_experts=total_expert_count,
        top_k=routing_top_k,
    )

    request_arrivals = generate_request_arrivals(
        total_requests=total_requests,
        arrival_rate=arrival_rate,
        rng=rng,
    )

    simulator = SyncMoESimulator(
        request_arrivals=request_arrivals,
        tokens_per_request=tokens_per_request,
        num_layers=num_layers,
        total_expert_count=total_expert_count,
        ep_group_size=ep_group_size,
        max_batch_size=max_batch_size,
        attn_service_t=attn_service_t,
        net_t_attn_to_expert=net_t_attn_to_expert,
        net_t_expert_to_attn=net_t_expert_to_attn,
        compute_time_lookup=compute_time_lookup,
        profile_router=profile_router,
        progress_tracker=progress_tracker,
        global_request_max_batch_size=global_request_max_batch_size,
    )

    results = simulator.run()
    completion_times = results["completion_times"]
    token_latencies = results["token_latencies"]
    request_latency_values = results["request_latency_values"]
    makespan = results["makespan"]
    avg_layer_expert_runtime = results["avg_layer_expert_runtime"]
    avg_layer_wait_imbalance = results["avg_layer_wait_imbalance"]

    if len(completion_times) != total_tokens_expected:
        print(
            "WARNING: some tokens did not complete!",
            len(completion_times),
            "/",
            total_tokens_expected,
        )

    latencies = list(token_latencies.values())
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    makespan_ms = makespan / ticks_per_millisecond
    makespan_sec = makespan_ms / 1000.0 if makespan_ms > 0 else 0.0
    avg_throughput_req_per_sec = (
        total_requests / makespan_sec if makespan_sec > 0 else 0.0
    )

    request_latencies_ms = [
        latency / ticks_per_millisecond for latency in request_latency_values
    ]
    avg_request_latency_ms = (
        sum(request_latencies_ms) / len(request_latencies_ms)
        if request_latencies_ms
        else 0.0
    )
    p90_request_latency_ms = (
        percentile(request_latency_values, 0.90) / ticks_per_millisecond
        if request_latency_values
        else 0.0
    )
    p99_request_latency_ms = (
        percentile(request_latency_values, 0.99) / ticks_per_millisecond
        if request_latency_values
        else 0.0
    )

    avg_layer_expert_runtime_ms = avg_layer_expert_runtime / ticks_per_millisecond
    avg_layer_wait_imbalance_ms = avg_layer_wait_imbalance / ticks_per_millisecond

    print(f"Simulation finished at time {makespan:.3f}")
    print(f"Average completion time over {len(latencies)} tokens: {avg_latency:.3f}")
    print(f"Average throughput: {avg_throughput_req_per_sec:.3f} requests/sec")
    print(f"Average request latency: {avg_request_latency_ms:.3f} ms")
    print(f"P90 request latency: {p90_request_latency_ms:.3f} ms")
    print(f"P99 request latency: {p99_request_latency_ms:.3f} ms")
    print(f"Average per-layer expert runtime: {avg_layer_expert_runtime_ms:.3f} ms")
    print(f"Average per-layer longest wait: {avg_layer_wait_imbalance_ms:.3f} ms")

    return {
        "completion_times": completion_times,
        "avg_latency": avg_latency,
        "makespan": makespan,
        "avg_layer_expert_runtime": avg_layer_expert_runtime,
        "avg_layer_wait_imbalance": avg_layer_wait_imbalance,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
    }


if __name__ == "__main__":
    run_simulation()
