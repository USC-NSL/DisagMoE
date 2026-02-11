from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Tuple

import torch

from util import expert_compute_time_lookup_table_from_profile, build_profile_router
from disagmoe.models.gate import ProfileDrivenRouter


# ------------------------------
# Configurable parameters
# ------------------------------

EP_GROUP_SIZE = 16  # "n": number of expert workers globally
TOTAL_EXPERT_COUNT = 128  # total experts per layer
MAX_BATCH_SIZE = 512
NUM_LAYERS = 48  # number of expert layers
TOTAL_REQUESTS = 8192
TOKENS_PER_REQUEST = 256
TOTAL_TOKENS = TOTAL_REQUESTS * TOKENS_PER_REQUEST
GLOBAL_REQUEST_MAX_BATCH_SIZE = EP_GROUP_SIZE * 256  # max concurrent active requests

ARRIVAL_RATE = 50.0  # lambda for Poisson arrivals (requests / tick)

ATTN_SERVICE_T = 2  # time (ticks) per token at attention worker
ATTN_DP_GROUP_SIZE = EP_GROUP_SIZE  # number of parallel attention workers globally
TICKS_PER_MILLISECOND = 10  # 0.1 ms per tick

NET_T_EXPERT_TO_ATTN = 0.1  # fixed network delay expert -> attention in #ticks, 10us
NET_T_ATTN_TO_EXPERT = 0.1  # fixed network delay attention -> expert in #ticks, 10us

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
    # Models the attention (DP) GPU assigned to this request/token. Used for
    # topology-aware network delay modeling.
    home_gpu: int = 0
    layer_fanout: Dict[int, int] = field(default_factory=dict)
    sampled_for_stats: bool = False


@dataclass
class RequestState:
    request_id: int
    arrival_time: float
    home_gpu: int | None = None
    next_token_index: int = 0
    completed: bool = False


@dataclass(frozen=True)
class TimelineOp:
    op_idx: int
    step_idx: int
    microbatch: int
    layer_idx: int
    stage: str  # attn | dispatch | expert | return
    resource: str  # attn | comm | expert
    start_time: float  # absolute ticks
    end_time: float  # absolute ticks


class TimelineCapture:
    def __init__(self, max_ops: int):
        self.max_ops = max(0, int(max_ops))
        self.started = False
        self.start_step: int | None = None
        self.start_time: float | None = None
        self.ops: List[TimelineOp] = []

    @property
    def enabled(self) -> bool:
        return self.max_ops > 0

    def maybe_start(self, *, step_idx: int, sim_time: float) -> None:
        if not self.enabled or self.started:
            return
        self.started = True
        self.start_step = int(step_idx)
        self.start_time = float(sim_time)

    def record(
        self,
        *,
        step_idx: int,
        microbatch: int,
        layer_idx: int,
        stage: str,
        resource: str,
        start_time: float,
        end_time: float,
    ) -> None:
        if not self.started or not self.enabled:
            return
        if len(self.ops) >= self.max_ops:
            return
        self.ops.append(
            TimelineOp(
                op_idx=len(self.ops),
                step_idx=int(step_idx),
                microbatch=int(microbatch),
                layer_idx=int(layer_idx),
                stage=str(stage),
                resource=str(resource),
                start_time=float(start_time),
                end_time=float(end_time),
            )
        )

    def write_csv(self, out_path: str, *, ticks_per_millisecond: float) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "op_idx,step_idx,microbatch,layer_idx,stage,resource,"
                "start_ticks,end_ticks,duration_ticks,start_ms,end_ms,duration_ms\n"
            )
            for op in self.ops:
                dur = op.end_time - op.start_time
                start_ms = (
                    op.start_time / float(ticks_per_millisecond)
                    if ticks_per_millisecond
                    else 0.0
                )
                end_ms = (
                    op.end_time / float(ticks_per_millisecond)
                    if ticks_per_millisecond
                    else 0.0
                )
                dur_ms = dur / float(ticks_per_millisecond) if ticks_per_millisecond else 0.0
                f.write(
                    f"{op.op_idx},"
                    f"{op.step_idx},"
                    f"{op.microbatch},"
                    f"{op.layer_idx},"
                    f"{op.stage},"
                    f"{op.resource},"
                    f"{op.start_time:.6f},"
                    f"{op.end_time:.6f},"
                    f"{dur:.6f},"
                    f"{start_ms:.6f},"
                    f"{end_ms:.6f},"
                    f"{dur_ms:.6f}\n"
                )


def _plot_tbo_microbatch_timeline(
    capture: TimelineCapture,
    out_path: str,
    *,
    ticks_per_millisecond: float,
    title: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not capture.ops:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    base = min(op.start_time for op in capture.ops)

    def to_ms(ticks: float) -> float:
        return ticks / float(ticks_per_millisecond) if ticks_per_millisecond else 0.0

    color_by_stage = {
        "attn": "#1f77b4",
        "dispatch": "#ff7f0e",
        "expert": "#2ca02c",
        "combine": "#d62728",
    }

    rows = {0: 10, 1: 0}
    height = 8

    fig, ax = plt.subplots(figsize=(14, 3.8))
    for mb in (0, 1):
        ops = [op for op in capture.ops if op.microbatch == mb]
        for op in ops:
            left = to_ms(op.start_time - base)
            width = max(0.0, to_ms(op.end_time - op.start_time))
            ax.broken_barh(
                [(left, width)],
                (rows[mb], height),
                facecolors=color_by_stage.get(op.stage, "#7f7f7f"),
                edgecolors="black",
                linewidth=0.4,
                alpha=0.9,
            )

    ax.set_yticks([rows[0] + height / 2.0, rows[1] + height / 2.0])
    ax.set_yticklabels(["microbatch 0", "microbatch 1"])
    ax.set_xlabel("time (ms, relative)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

    legend_handles = []
    legend_labels = []
    for stage in ("attn", "dispatch", "expert", "combine"):
        legend_handles.append(plt.Line2D([0], [0], color=color_by_stage[stage], linewidth=8))
        legend_labels.append(stage)
    ax.legend(legend_handles, legend_labels, ncol=4, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


class ProgressTracker:
    """Simple stdout progress bar with ETA estimation."""

    def __init__(self, total_tokens: int, bar_width: int = 30, min_interval: float = 1.0):
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
    The first request arrives at time 0.
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


class TBOMoESimulator:
    """
    Centralized synchronous MoE simulator with 2-way microbatch pipelining to overlap
    communication and compute ("tbo" = two-batch overlap).

    Each iteration spawns one token per active request, splits tokens evenly into
    two microbatches, and then runs a centralized schedule with:
      - compute: attention (DP) compute and expert (EP) compute (single-capacity, mutually exclusive)
      - comm: dispatch/combine latency modeled as asynchronous in-flight transfer

    Note: This model allows overlapping communication with compute, but does NOT
    allow overlapping attention compute with expert compute. Combine only blocks
    if its in-flight transfer has not finished at the time the next compute stage
    needs the data.
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
        attn_dp_group_size: int,
        net_t_attn_to_expert: float,
        net_t_expert_to_attn: float,
        compute_time_lookup: Dict[int, float],
        profile_router: ProfileDrivenRouter,
        progress_tracker: ProgressTracker,
        global_request_max_batch_size: int,
        *,
        net_delay_fn: Callable[[int, int, int], float] | None = None,
        n_gpu_per_host: int | None = None,
        per_token_begin_cb: Callable[[Token], None] | None = None,
        per_token_stats_cb: Callable[[Token, float], None] | None = None,
        timeline_capture: TimelineCapture | None = None,
    ):
        if total_expert_count % ep_group_size != 0:
            raise ValueError("total_expert_count must be divisible by ep_group_size")

        self.tokens_per_request = tokens_per_request
        self.num_layers = num_layers
        self.total_expert_count = total_expert_count
        self.ep_group_size = ep_group_size
        self.max_batch_size = max_batch_size
        self.attn_service_t = float(attn_service_t)
        self.attn_dp_group_size = max(1, int(attn_dp_group_size))
        self.net_t_attn_to_expert = float(net_t_attn_to_expert)
        self.net_t_expert_to_attn = float(net_t_expert_to_attn)
        self.net_delay_fn = net_delay_fn
        self.n_gpu_per_host = max(1, int(n_gpu_per_host)) if n_gpu_per_host is not None else None
        self.compute_time_lookup = compute_time_lookup
        self.profile_router = profile_router
        self.progress_tracker = progress_tracker
        self.global_request_max_batch_size = max(1, int(global_request_max_batch_size))
        self.per_token_begin_cb = per_token_begin_cb
        self.per_token_stats_cb = per_token_stats_cb
        self.timeline_capture = timeline_capture

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
        self._next_home_gpu = 0
        self.completion_times: Dict[int, float] = {}
        self.token_latencies: Dict[int, float] = {}
        self.request_latency_values: List[float] = []
        self.completed_tokens = 0

        self.queues_per_worker = self.total_expert_count // self.ep_group_size
        self._router_device = torch.device("cpu")
        self._router_dtype = torch.float32

        # Per-layer metrics for imbalance analysis (per microbatch)
        self.layer_expert_runtimes: List[float] = []
        self.layer_wait_imbalances: List[float] = []
        self.worker_queue_stddevs: List[float] = []

        # Batch size statistics (per expert compute invocation)
        self.total_expert_batch_size: float = 0.0
        self.total_expert_batch_count: int = 0

    def run(self, tokens_after_reaching_max_bs: int | None = None):
        steps_executed = 0
        saturation_step: int | None = None
        saturation_tokens_completed: int | None = None
        saturation_time: float | None = None
        capture = self.timeline_capture

        if tokens_after_reaching_max_bs is not None:
            tokens_after_reaching_max_bs = int(tokens_after_reaching_max_bs)
            if tokens_after_reaching_max_bs < 0:
                raise ValueError("tokens_after_reaching_max_bs must be >= 0 or None")

            while self.completed_tokens < self.expected_total_tokens:
                steps_executed += 1

                self._release_new_arrivals()
                self._fill_active_requests()

                if (
                    saturation_step is None
                    and self.active_requests
                    and len(self.active_requests) >= self.global_request_max_batch_size
                ):
                    saturation_step = steps_executed
                    saturation_tokens_completed = self.completed_tokens
                    saturation_time = self.current_time
                    if capture is not None:
                        capture.maybe_start(step_idx=steps_executed, sim_time=self.current_time)

                if not self.active_requests:
                    if not self._advance_to_next_arrival():
                        break
                else:
                    tokens = self._spawn_tokens_for_iteration()
                    if tokens:
                        start_time = self.current_time
                        iteration_duration, token_completion_times = self._run_one_iteration(
                            tokens,
                            step_idx=steps_executed,
                            iteration_base_time=start_time,
                            timeline_capture=capture,
                        )
                        self.current_time = start_time + iteration_duration
                        self._finalize_iteration(tokens, token_completion_times)
                    else:
                        if not self._advance_to_next_arrival():
                            break

                if (
                    saturation_tokens_completed is not None
                    and (self.completed_tokens - saturation_tokens_completed)
                    >= tokens_after_reaching_max_bs
                ):
                    break
        else:
            while self.completed_tokens < self.expected_total_tokens:
                steps_executed += 1
                self._release_new_arrivals()
                self._fill_active_requests()

                if (
                    saturation_step is None
                    and self.active_requests
                    and len(self.active_requests) >= self.global_request_max_batch_size
                ):
                    saturation_step = steps_executed
                    saturation_tokens_completed = self.completed_tokens
                    saturation_time = self.current_time
                    if capture is not None:
                        capture.maybe_start(step_idx=steps_executed, sim_time=self.current_time)

                if not self.active_requests:
                    if not self._advance_to_next_arrival():
                        break
                    continue

                tokens = self._spawn_tokens_for_iteration()
                if not tokens:
                    if not self._advance_to_next_arrival():
                        break
                    continue

                start_time = self.current_time
                iteration_duration, token_completion_times = self._run_one_iteration(
                    tokens,
                    step_idx=steps_executed,
                    iteration_base_time=start_time,
                    timeline_capture=capture,
                )
                self.current_time = start_time + iteration_duration
                self._finalize_iteration(tokens, token_completion_times)

        if self.completed_tokens >= self.expected_total_tokens:
            self.progress_tracker.finalize()
        else:
            self.progress_tracker.update(self.completed_tokens, force=True)
            sys.stdout.write("\n")

        if self.layer_expert_runtimes:
            avg_layer_runtime = sum(self.layer_expert_runtimes) / len(self.layer_expert_runtimes)
        else:
            avg_layer_runtime = 0.0

        if self.layer_wait_imbalances:
            avg_layer_wait_imbalance = sum(self.layer_wait_imbalances) / len(self.layer_wait_imbalances)
        else:
            avg_layer_wait_imbalance = 0.0

        if self.worker_queue_stddevs:
            avg_layer_worker_queue_stddev = sum(self.worker_queue_stddevs) / len(self.worker_queue_stddevs)
        else:
            avg_layer_worker_queue_stddev = 0.0

        if self.total_expert_batch_count > 0:
            avg_per_expert_batch_size = self.total_expert_batch_size / self.total_expert_batch_count
        else:
            avg_per_expert_batch_size = 0.0

        stopped_early = self.completed_tokens != self.expected_total_tokens

        return {
            "completion_times": self.completion_times,
            "token_latencies": self.token_latencies,
            "request_latency_values": self.request_latency_values,
            "makespan": self.current_time,
            "avg_layer_runtime": avg_layer_runtime,
            "avg_layer_wait_imbalance": avg_layer_wait_imbalance,
            "avg_layer_worker_queue_stddev": avg_layer_worker_queue_stddev,
            "avg_per_expert_batch_size": avg_per_expert_batch_size,
            "stopped_early": stopped_early,
            "steps_executed": steps_executed,
            "saturation_step": saturation_step,
            "saturation_tokens_completed": saturation_tokens_completed,
            "saturation_time": saturation_time,
            "timeline_capture_started": bool(capture.started) if capture is not None else False,
            "timeline_ops_recorded": len(capture.ops) if capture is not None else 0,
        }

    # ------------------------------
    # Internal helpers (same lifecycle as sim_sync)
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
            if candidate.home_gpu is None:
                candidate.home_gpu = (self._next_home_gpu % self.attn_dp_group_size)
                self._next_home_gpu += 1
            self.active_requests.append(candidate)

    def _spawn_tokens_for_iteration(self) -> List[Token]:
        tokens: List[Token] = []
        begin_cb = self.per_token_begin_cb
        for request in self.active_requests:
            if request.completed or request.next_token_index >= self.tokens_per_request:
                continue
            if request.home_gpu is None:
                raise RuntimeError(
                    f"Request {request.request_id} is active but missing home_gpu assignment."
                )
            token = Token(
                tid=self.next_tid,
                birth_time=self.current_time,
                request_id=request.request_id,
                token_index=request.next_token_index,
                home_gpu=int(request.home_gpu),
            )
            self.next_tid += 1
            if begin_cb is not None:
                begin_cb(token)
            tokens.append(token)
        return tokens

    def _split_into_microbatches(self, tokens: List[Token]) -> Tuple[List[Token], List[Token], Dict[int, int]]:
        if not tokens:
            return [], [], {}
        tokens_sorted = sorted(tokens, key=lambda t: (t.request_id, t.tid))
        mb0: List[Token] = []
        mb1: List[Token] = []
        tid_to_mb: Dict[int, int] = {}
        for i, token in enumerate(tokens_sorted):
            mb = 0 if (i % 2 == 0) else 1
            tid_to_mb[token.tid] = mb
            (mb0 if mb == 0 else mb1).append(token)
        return mb0, mb1, tid_to_mb

    def _run_one_iteration(
        self,
        tokens: List[Token],
        *,
        step_idx: int,
        iteration_base_time: float,
        timeline_capture: TimelineCapture | None,
    ) -> tuple[float, Dict[int, float]]:
        mb0, mb1, tid_to_mb = self._split_into_microbatches(tokens)

        # Pre-build tasks by layer; for each layer route once over all tokens,
        # then compute per-microbatch stage times.
        # Task kinds:
        # - compute: consumes the single-capacity compute resource
        # - dispatch/combine: launches an async transfer (in-flight), may block later if not finished
        tasks: List[List[tuple[str, float, str, int]]] = [[], []]  # (kind, dur, stage, layer_idx) per microbatch

        for layer_idx in range(self.num_layers):
            layer_times = self._compute_layer_times(layer_idx, tokens, tid_to_mb)
            # Each microbatch does: attn (compute) -> dispatch (comm, async) -> expert (compute) -> combine (comm, async)
            for mb in (0, 1):
                attn_t, dispatch_t, expert_compute_t, combine_t = layer_times[mb]
                tasks[mb].append(("compute", attn_t, "attn", layer_idx))
                tasks[mb].append(("dispatch", dispatch_t, "dispatch", layer_idx))
                tasks[mb].append(("compute", expert_compute_t, "expert", layer_idx))
                tasks[mb].append(("combine", combine_t, "combine", layer_idx))

        finish0, finish1 = self._schedule_resources(
            tasks[0],
            tasks[1],
            step_idx=int(step_idx),
            iteration_base_time=float(iteration_base_time),
            timeline_capture=timeline_capture,
        )
        makespan = max(finish0, finish1)

        token_completion_times: Dict[int, float] = {}
        for token in tokens:
            mb = tid_to_mb.get(token.tid, 0)
            token_completion_times[token.tid] = (finish0 if mb == 0 else finish1)

        return makespan, token_completion_times

    def _is_same_host(self, gpu_a: int, gpu_b: int) -> bool:
        if self.n_gpu_per_host is None:
            return True
        return (int(gpu_a) // self.n_gpu_per_host) == (int(gpu_b) // self.n_gpu_per_host)

    def _compute_layer_times(
        self,
        layer_idx: int,
        tokens: List[Token],
        tid_to_mb: Dict[int, int],
    ) -> List[tuple[float, float, float, float]]:
        # Returns per-microbatch (attn, dispatch, expert_compute, combine_comm)
        token_count_mb = [0, 0]
        for token in tokens:
            token_count_mb[tid_to_mb[token.tid]] += 1

        attn_times = []
        for mb in (0, 1):
            n = token_count_mb[mb]
            if n <= 0 or self.attn_service_t <= 0.0:
                attn_times.append(0.0)
            else:
                batches = math.ceil(n / float(self.attn_dp_group_size))
                attn_times.append(batches * self.attn_service_t)

        # Route once for all tokens; then aggregate per microbatch.
        worker_queues_by_mb: List[List[List[int]]] = [
            [[0 for _ in range(self.queues_per_worker)] for _ in range(self.ep_group_size)]
            for _ in range(2)
        ]
        worker_src_gpus_by_mb: List[List[set[int]]] = [
            [set() for _ in range(self.ep_group_size)] for _ in range(2)
        ]
        # tokens_per_src_per_dst_by_mb[mb][dst_worker][src_gpu] = token count
        tokens_per_src_per_dst_by_mb: List[Dict[int, Dict[int, int]]] = [{}, {}]

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
            mb = tid_to_mb[token.tid]
            src_gpu = int(token.home_gpu)

            for expert_id in selected:
                if expert_id >= self.total_expert_count:
                    raise RuntimeError(
                        f"Expert id {expert_id} exceeds total experts {self.total_expert_count}"
                    )
                worker_idx = expert_id // self.queues_per_worker
                local_queue_idx = expert_id % self.queues_per_worker
                worker_queues_by_mb[mb][worker_idx][local_queue_idx] += 1
                worker_src_gpus_by_mb[mb][worker_idx].add(src_gpu)
                # Track per-src per-dst token counts for congestion modeling.
                dst_map = tokens_per_src_per_dst_by_mb[mb].get(worker_idx)
                if dst_map is None:
                    dst_map = {}
                    tokens_per_src_per_dst_by_mb[mb][worker_idx] = dst_map
                dst_map[src_gpu] = dst_map.get(src_gpu, 0) + 1

        out: List[tuple[float, float, float, float]] = []
        for mb in (0, 1):
            dispatch_time = self.net_t_attn_to_expert
            tokens_per_src_per_dst = tokens_per_src_per_dst_by_mb[mb]

            if self.net_delay_fn is not None:
                # --- Congestion-aware dispatch (dual receive-queue model) ---
                # Each destination GPU has two independent receive queues:
                # NVLink (intra-node) and RDMA (inter-node).  Transfers are
                # serialized within each queue at full link bandwidth, but
                # the two queues drain in parallel.
                dst_dispatch_delays: List[float] = []
                for worker_idx in range(self.ep_group_size):
                    src_tokens = tokens_per_src_per_dst.get(worker_idx)
                    if not src_tokens:
                        dst_dispatch_delays.append(0.0)
                        continue
                    nvlink_total = 0.0
                    rdma_total = 0.0
                    for src, ntok in src_tokens.items():
                        if ntok <= 0:
                            continue
                        delay = float(self.net_delay_fn(int(src), int(worker_idx), int(ntok)))
                        if self._is_same_host(src, worker_idx):
                            nvlink_total += delay
                        else:
                            rdma_total += delay
                    dst_dispatch_delays.append(max(nvlink_total, rdma_total))
                dispatch_time = max(dst_dispatch_delays) if dst_dispatch_delays else 0.0

            # Record per-worker queue imbalance (stddev over that worker's expert queues).
            for queues in worker_queues_by_mb[mb]:
                if not queues:
                    continue
                mean = sum(queues) / float(len(queues))
                variance = sum((q - mean) ** 2 for q in queues) / float(len(queues))
                self.worker_queue_stddevs.append(math.sqrt(variance))

            # Compute expert processing time (compute only, no return delay).
            worker_compute_times: List[float] = []
            worker_total_tokens: List[int] = []
            for worker_idx, loads in enumerate(worker_queues_by_mb[mb]):
                compute_t = self._simulate_worker_compute_time(loads)
                worker_compute_times.append(compute_t)
                worker_total_tokens.append(sum(loads))
            expert_compute_time = max(worker_compute_times) if worker_compute_times else 0.0

            # --- Congestion-aware return (dual receive-queue model) ---
            # Each attention GPU (destination) has NVLink and RDMA receive
            # queues that drain independently.
            if self.net_delay_fn is not None:
                attn_nvlink_delays: Dict[int, float] = {}
                attn_rdma_delays: Dict[int, float] = {}
                for worker_idx in range(self.ep_group_size):
                    if worker_total_tokens[worker_idx] == 0:
                        continue
                    src_tokens = tokens_per_src_per_dst.get(worker_idx)
                    if not src_tokens:
                        continue
                    for attn_gpu in src_tokens:
                        tokens_to_return = src_tokens[attn_gpu]
                        if tokens_to_return <= 0:
                            continue
                        delay = float(self.net_delay_fn(int(worker_idx), int(attn_gpu), int(tokens_to_return)))
                        if self._is_same_host(worker_idx, attn_gpu):
                            attn_nvlink_delays[attn_gpu] = attn_nvlink_delays.get(attn_gpu, 0.0) + delay
                        else:
                            attn_rdma_delays[attn_gpu] = attn_rdma_delays.get(attn_gpu, 0.0) + delay
                all_attn_gpus = set(attn_nvlink_delays.keys()) | set(attn_rdma_delays.keys())
                if all_attn_gpus:
                    return_comm_time = max(
                        max(attn_nvlink_delays.get(gpu, 0.0), attn_rdma_delays.get(gpu, 0.0))
                        for gpu in all_attn_gpus
                    )
                else:
                    return_comm_time = 0.0
            else:
                return_comm_time = self.net_t_expert_to_attn

            # Per-layer wait imbalance: spread of worker compute runtimes.
            if worker_compute_times:
                earliest_finish = min(worker_compute_times)
                latest_finish = max(worker_compute_times)
                longest_wait = max(0.0, latest_finish - earliest_finish)
            else:
                longest_wait = 0.0

            layer_runtime = attn_times[mb] + dispatch_time + expert_compute_time + return_comm_time
            self.layer_expert_runtimes.append(layer_runtime)
            self.layer_wait_imbalances.append(longest_wait)

            out.append((attn_times[mb], dispatch_time, expert_compute_time, return_comm_time))

        return out

    def _simulate_worker_compute_time(self, loads: List[int]) -> float:
        total_tokens = sum(loads)
        if total_tokens <= 0:
            return 0.0

        queues = list(loads)
        elapsed = 0.0
        while total_tokens > 0:
            queue_idx = max(range(len(queues)), key=lambda i: queues[i])
            available = queues[queue_idx]
            if available <= 0:
                break
            batch = min(available, self.max_batch_size)
            queues[queue_idx] -= batch
            total_tokens -= batch
            compute_t = float(self.compute_time_lookup[batch])
            elapsed += compute_t
            self.total_expert_batch_size += batch
            self.total_expert_batch_count += 1
        return elapsed

    @staticmethod
    def _schedule_resources(
        tasks0: List[tuple],
        tasks1: List[tuple],
        *,
        step_idx: int,
        iteration_base_time: float,
        timeline_capture: TimelineCapture | None,
    ) -> tuple[float, float]:
        tasks = [tasks0, tasks1]
        next_idx = [0, 0]
        ready = [0.0, 0.0]
        compute_free = 0.0
        inflight_until = [0.0, 0.0]  # per-microbatch async comm completion time
        inflight_src: list[tuple[str, int] | None] = [None, None]  # (stage, layer_idx)

        def kind_prio(kind: str) -> int:
            # Deterministic tie-break: prefer launching comm early at the same timestamp.
            if kind in ("dispatch", "combine"):
                return 0
            return 1

        while True:
            candidates: List[tuple[float, int, int]] = []
            for mb in (0, 1):
                if next_idx[mb] >= len(tasks[mb]):
                    continue
                t = tasks[mb][next_idx[mb]]
                kind = str(t[0])
                if kind == "compute":
                    est = max(ready[mb], inflight_until[mb], compute_free)
                elif kind in ("dispatch", "combine"):
                    # Launch is async and doesn't consume a serialized resource.
                    est = ready[mb]
                else:
                    raise ValueError(f"Unknown task kind: {kind}")
                candidates.append((est, kind_prio(kind), mb))

            if not candidates:
                break

            est, _prio, mb = min(candidates)
            t = tasks[mb][next_idx[mb]]
            kind = str(t[0])
            dur = float(t[1])
            stage = str(t[2]) if len(t) >= 3 else kind
            layer_idx = int(t[3]) if len(t) >= 4 else -1

            if kind == "compute":
                # If compute is waiting solely due to an unfinished async comm,
                # record that wait (comm only appears if it blocks).
                t0 = max(ready[mb], compute_free)
                if inflight_until[mb] > t0:
                    src = inflight_src[mb]
                    wait_stage, wait_layer = (src if src is not None else ("comm", -1))
                    if (
                        timeline_capture is not None
                        and timeline_capture.started
                        and len(timeline_capture.ops) < timeline_capture.max_ops
                    ):
                        timeline_capture.record(
                            step_idx=int(step_idx),
                            microbatch=int(mb),
                            layer_idx=int(wait_layer),
                            stage=str(wait_stage),
                            resource="comm",
                            start_time=float(iteration_base_time) + float(t0),
                            end_time=float(iteration_base_time) + float(inflight_until[mb]),
                        )

                start = est
                end = start + dur
                ready[mb] = end
                compute_free = end

                if (
                    timeline_capture is not None
                    and timeline_capture.started
                    and len(timeline_capture.ops) < timeline_capture.max_ops
                ):
                    timeline_capture.record(
                        step_idx=int(step_idx),
                        microbatch=int(mb),
                        layer_idx=int(layer_idx),
                        stage=stage,
                        resource="compute",
                        start_time=float(iteration_base_time) + float(start),
                        end_time=float(iteration_base_time) + float(end),
                    )
            else:
                # Async launch: doesn't advance ready time, but starts an in-flight transfer.
                start = float(est)
                end = start + dur
                if end > inflight_until[mb]:
                    inflight_until[mb] = end
                    inflight_src[mb] = (stage, int(layer_idx))

            next_idx[mb] += 1

        return ready[0], ready[1]

    def _finalize_iteration(self, tokens: List[Token], token_completion_offsets: Dict[int, float]):
        cb = self.per_token_stats_cb

        for token in tokens:
            completion_time = token.birth_time + float(token_completion_offsets.get(token.tid, 0.0))
            self.completion_times[token.tid] = completion_time
            latency = completion_time - token.birth_time
            self.token_latencies[token.tid] = latency
            self.completed_tokens += 1

            if cb is not None and bool(getattr(token, "sampled_for_stats", False)):
                cb(token, completion_time)

            req = self.request_lookup[token.request_id]
            req.next_token_index += 1
            if req.next_token_index >= self.tokens_per_request:
                req.completed = True
                self.request_latency_values.append(completion_time - req.arrival_time)

        self.progress_tracker.update(self.completed_tokens)
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
    attn_dp_group_size=ATTN_DP_GROUP_SIZE,
    net_t_attn_to_expert=NET_T_ATTN_TO_EXPERT,
    net_t_expert_to_attn=NET_T_EXPERT_TO_ATTN,
    net_delay_fn: Callable[[int, int, int], float] | None = None,
    n_gpu_per_host: int | None = None,
    ticks_per_millisecond=TICKS_PER_MILLISECOND,
    expert_profile_path=EXPERT_COMPUTE_PROFILE_PATH,
    profile_routing_path=PROFILE_ROUTING_PATH,
    routing_top_k=ROUTING_TOP_K,
    rng_seed=RNG_SEED,
    global_request_max_batch_size=GLOBAL_REQUEST_MAX_BATCH_SIZE,
    tokens_after_reaching_max_bs: int | None = None,
    per_token_begin_cb: Callable[[Token], None] | None = None,
    per_token_stats_cb: Callable[[Token, float], None] | None = None,
    timeline_capture_ops: int = 0,
    timeline_out_dir: str | None = None,
    timeline_basename: str | None = None,
):
    rng = random.Random(rng_seed)

    total_tokens_expected = total_requests * tokens_per_request
    progress_tracker = ProgressTracker(total_tokens_expected)

    base_compute_time_lookup = expert_compute_time_lookup_table_from_profile(
        expert_profile_path,
        max_batch_size,
        ticks_per_millisecond=ticks_per_millisecond,
    )
    compute_time_lookup: Dict[int, float] = {
        bs: t * GROUP_GEMM_SPEEDUP_FACTOR for bs, t in base_compute_time_lookup.items()
    }

    profile_router = build_profile_router(
        profile_path=profile_routing_path,
        num_experts=total_expert_count,
        top_k=routing_top_k,
    )

    attn_dp_group_size = max(1, int(attn_dp_group_size))

    request_arrivals = generate_request_arrivals(
        total_requests=total_requests,
        arrival_rate=arrival_rate,
        rng=rng,
    )

    timeline_capture = (
        TimelineCapture(int(timeline_capture_ops))
        if int(timeline_capture_ops) > 0
        else None
    )
    simulator = TBOMoESimulator(
        request_arrivals=request_arrivals,
        tokens_per_request=tokens_per_request,
        num_layers=num_layers,
        total_expert_count=total_expert_count,
        ep_group_size=ep_group_size,
        max_batch_size=max_batch_size,
        attn_service_t=attn_service_t,
        attn_dp_group_size=attn_dp_group_size,
        net_t_attn_to_expert=net_t_attn_to_expert,
        net_t_expert_to_attn=net_t_expert_to_attn,
        net_delay_fn=net_delay_fn,
        n_gpu_per_host=n_gpu_per_host,
        compute_time_lookup=compute_time_lookup,
        profile_router=profile_router,
        progress_tracker=progress_tracker,
        global_request_max_batch_size=global_request_max_batch_size,
        per_token_begin_cb=per_token_begin_cb,
        per_token_stats_cb=per_token_stats_cb,
        timeline_capture=timeline_capture,
    )

    results = simulator.run(tokens_after_reaching_max_bs=tokens_after_reaching_max_bs)
    if timeline_capture is not None and timeline_capture.started and timeline_capture.ops:
        out_dir = (
            str(timeline_out_dir)
            if timeline_out_dir is not None
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "tbo-timeline")
        )
        os.makedirs(out_dir, exist_ok=True)
        if timeline_basename is None:
            timeline_basename = (
                f"tbo_timeline_ep{ep_group_size}_gbs{global_request_max_batch_size}_attn{attn_service_t}_pid{os.getpid()}"
            )
        csv_path = os.path.join(out_dir, f"{timeline_basename}.csv")
        png_path = os.path.join(out_dir, f"{timeline_basename}.png")
        timeline_capture.write_csv(csv_path, ticks_per_millisecond=float(ticks_per_millisecond))
        title = (
            f"TBO timeline (first {len(timeline_capture.ops)}/{timeline_capture.max_ops} ops after saturation)\\n"
            f"ep_group_size={ep_group_size} global_bs={global_request_max_batch_size} attn_t={attn_service_t}"
        )
        plotted = _plot_tbo_microbatch_timeline(
            timeline_capture,
            png_path,
            ticks_per_millisecond=float(ticks_per_millisecond),
            title=title,
        )
        results["timeline_csv_path"] = csv_path
        results["timeline_png_path"] = png_path if plotted else None
        print(f"[tbo timeline] wrote csv: {csv_path}")
        if plotted:
            print(f"[tbo timeline] wrote png: {png_path}")
        else:
            print("[tbo timeline] matplotlib unavailable; skipped png plot (csv still written)")
    elif timeline_capture is not None and timeline_capture.enabled:
        results["timeline_csv_path"] = None
        results["timeline_png_path"] = None
    completion_times = results["completion_times"]
    token_latencies = results["token_latencies"]
    request_latency_values = results["request_latency_values"]
    makespan = results["makespan"]
    avg_layer_runtime = results["avg_layer_runtime"]
    avg_layer_wait_imbalance = results["avg_layer_wait_imbalance"]
    avg_layer_worker_queue_stddev = results["avg_layer_worker_queue_stddev"]
    avg_per_expert_batch_size = results["avg_per_expert_batch_size"]
    stopped_early = results.get("stopped_early", False)
    steps_executed = results.get("steps_executed", 0)
    saturation_step = results.get("saturation_step", None)
    saturation_time = results.get("saturation_time", None)
    saturation_tokens_completed = results.get("saturation_tokens_completed", None)

    if stopped_early and tokens_after_reaching_max_bs is None:
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
    completed_tokens = len(completion_times)
    avg_throughput_req_per_sec = (
        (completed_tokens / float(tokens_per_request)) / makespan_sec
        if makespan_sec > 0 and tokens_per_request > 0
        else 0.0
    )

    tail_throughput_req_per_sec = float("nan")
    if (
        tokens_after_reaching_max_bs is not None
        and stopped_early
        and saturation_time is not None
        and saturation_tokens_completed is not None
        and tokens_per_request > 0
        and makespan > float(saturation_time)
    ):
        tail_tokens = completed_tokens - int(saturation_tokens_completed)
        tail_seconds = ((makespan - float(saturation_time)) / ticks_per_millisecond) / 1000.0
        if tail_seconds > 0:
            tail_throughput_req_per_sec = (tail_tokens / float(tokens_per_request)) / tail_seconds

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

    avg_layer_runtime_ms = avg_layer_runtime / ticks_per_millisecond
    avg_layer_wait_imbalance_ms = avg_layer_wait_imbalance / ticks_per_millisecond

    if stopped_early:
        print(
            f"Simulation stopped early at time {makespan:.3f} "
            f"(completed_tokens={completed_tokens}/{total_tokens_expected}, "
            f"steps_executed={steps_executed}, saturation_step={saturation_step})"
        )
    else:
        print(f"Simulation finished at time {makespan:.3f}")
    print(f"Average completion time over {len(latencies)} tokens: {avg_latency:.3f}")
    print(f"Average throughput: {avg_throughput_req_per_sec:.3f} requests/sec")
    if _is_finite(tail_throughput_req_per_sec):
        print(
            f"Tail throughput (after saturation, {tokens_after_reaching_max_bs} tokens): "
            f"{tail_throughput_req_per_sec:.3f} requests/sec"
        )
    print(f"Average request latency: {avg_request_latency_ms:.3f} ms")
    print(f"P90 request latency: {p90_request_latency_ms:.3f} ms")
    print(f"P99 request latency: {p99_request_latency_ms:.3f} ms")
    print(f"Average per-layer runtime: {avg_layer_runtime_ms:.3f} ms")
    print(f"Average per-layer longest wait: {avg_layer_wait_imbalance_ms:.3f} ms")
    print(f"Average per-expert batch size: {avg_per_expert_batch_size:.3f}")
    print(
        f"Average per-layer worker queue stddev: {avg_layer_worker_queue_stddev:.3f}"
    )

    return {
        "completion_times": completion_times,
        "avg_latency": avg_latency,
        "makespan": makespan,
        "avg_layer_runtime": avg_layer_runtime,
        "avg_layer_wait_imbalance": avg_layer_wait_imbalance,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
        "avg_layer_worker_queue_stddev": avg_layer_worker_queue_stddev,
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "stopped_early": stopped_early,
        "steps_executed": steps_executed,
        "saturation_step": saturation_step,
        "saturation_time": saturation_time,
        "timeline_capture_started": results.get("timeline_capture_started", False),
        "timeline_ops_recorded": results.get("timeline_ops_recorded", 0),
        "timeline_csv_path": results.get("timeline_csv_path", None),
        "timeline_png_path": results.get("timeline_png_path", None),
    }


def _is_finite(x: float) -> bool:
    return x == x and math.isfinite(x)


if __name__ == "__main__":
    run_simulation()
