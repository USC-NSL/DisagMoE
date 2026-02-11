from __future__ import annotations

import simpy
import random
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import math
import sys
import time
import torch

from util import (
    expert_compute_time_lookup_table_from_profile,
    build_profile_router,
)
from disagmoe.models.gate import ProfileDrivenRouter


class IngressPort:
    """Per-GPU receive port with two independent FIFO channels.

    Each GPU has one IngressPort containing:
      - An NVLink channel for intra-node transfers
      - An RDMA channel for inter-node transfers

    Each channel processes one transfer at a time at full link bandwidth.
    The two channels operate independently and can process in parallel.
    Incoming transfers are classified by topology (same host -> NVLink,
    different host -> RDMA) and queued on the appropriate channel.
    """

    def __init__(self, env: simpy.Environment, gpu_idx: int, n_gpu_per_host: int):
        self.env = env
        self.gpu_idx = int(gpu_idx)
        self.n_gpu_per_host = max(1, int(n_gpu_per_host))
        self._nvlink = simpy.Resource(env, capacity=1)
        self._rdma = simpy.Resource(env, capacity=1)

    def _is_same_host(self, other_gpu: int) -> bool:
        return (self.gpu_idx // self.n_gpu_per_host) == (int(other_gpu) // self.n_gpu_per_host)

    def transfer(self, src_gpu: int, solo_delay: float) -> simpy.events.Event:
        """Queue a transfer from *src_gpu* with the given full-bandwidth delay.

        The transfer is routed to the NVLink channel if *src_gpu* is on the
        same host as this GPU, or the RDMA channel otherwise.

        Returns a simpy event that triggers when this transfer completes.
        """
        if solo_delay <= 0:
            return self.env.timeout(0)

        is_intra = self._is_same_host(src_gpu)
        resource = self._nvlink if is_intra else self._rdma
        done_event = self.env.event()
        self.env.process(self._do_transfer(resource, solo_delay, done_event))
        return done_event

    def _do_transfer(self, resource: simpy.Resource, delay: float, done_event: simpy.events.Event):
        """FIFO transfer: acquire the channel, hold for *delay* ticks, release."""
        req = resource.request()
        yield req
        yield self.env.timeout(delay)
        resource.release(req)
        if not done_event.triggered:
            done_event.succeed()


# ------------------------------
# Configurable parameters
# ------------------------------

EP_GROUP_SIZE      = 16      # "n": number of expert workers globally
TOTAL_EXPERT_COUNT = 128     # total experts per layer
MAX_BATCH_SIZE     = 512
NUM_LAYERS         = 48      # number of expert layers
TOTAL_REQUESTS     = 8192
TOKENS_PER_REQUEST = 256
TOTAL_TOKENS       = TOTAL_REQUESTS * TOKENS_PER_REQUEST
GLOBAL_REQUEST_MAX_BATCH_SIZE = EP_GROUP_SIZE * 256  # max concurrent active requests

ARRIVAL_RATE       = 50.0   # lambda for Poisson arrivals (requests / tick)
# try for each attention worker, devide the arrival rate by the number of attention workers

ATTN_SERVICE_T     = 2    # time (ticks) per token at attention worker
ATTN_DP_GROUP_SIZE = EP_GROUP_SIZE  # number of parallel attention workers globally
TICKS_PER_MILLISECOND = 10  # 0.1 ms per tick

NET_T_EXPERT_TO_ATTN  = 0.1  # fixed network delay expert -> attention in #ticks, 10us
NET_T_ATTN_TO_EXPERT  = 0.1  # fixed network delay attention -> expert in #ticks, 10us

# Speedup factor for grouped GEMM vs single-expert profile.
# When batching all experts in a layer together (like the real system),
# grouped GEMM kernels are more efficient than processing experts one-by-one.
GROUP_GEMM_SPEEDUP_FACTOR = 0.7

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

# Expert scheduling policy used by each ExpertWorker.
#   "longest_queue_first" (default): pick the single longest queue.
#   "defragging_v0": approximate the C++ GroupLayerScheduler, looking
#                   ahead across layers for the same local expert id.
SCHEDULING_POLICY = "defragging_v0"

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
    # Stores how many experts each layer routed the token through.
    sampled_for_stats: bool = False

@dataclass
class DefragV0DebugState:
    enabled: bool = False
    triggered: bool = False
    trigger_time: float | None = None
    trigger_request_id: int | None = None
    trigger_active_requests: int | None = None
    trigger_pending_queue_len: int | None = None

    get_active_requests: Callable[[], int] | None = None
    get_pending_queue_len: Callable[[], int] | None = None
    get_global_expert_queued_tokens: Callable[[], int] | None = None
    get_global_expert_inflight_tokens: Callable[[], int] | None = None
    get_per_worker_expert_queued_tokens: Callable[[], List[int]] | None = None
    get_per_worker_expert_inflight_tokens: Callable[[], List[int]] | None = None
    get_attention_users: Callable[[], int] | None = None
    get_attention_queue: Callable[[], int] | None = None
    get_pending_tokens_total: Callable[[], int] | None = None
    get_pending_tokens_per_layer: Callable[[], List[int]] | None = None
    get_tokens_created: Callable[[], int] | None = None
    get_tokens_completed: Callable[[], int] | None = None

    logged: bool = False
    log_time: float | None = None
    worker_idx: int | None = None
    entry: Dict[str, object] | None = None


class RequestManager:
    """
    Manages sequential per-request token dispatch. A new request begins with
    token 0, and token i+1 is only injected after token i fully completes.
    """

    def __init__(self, env, tokens_per_request: int,
                 record_completion_cb: Callable[[Token, float], None],
                 request_complete_cb: Callable[[int, float], None] | None = None,
                 max_active_requests: int | None = None,
                 attn_dp_group_size: int = 1,
                 defrag_v0_debug_state: DefragV0DebugState | None = None,
                 token_begin_cb: Callable[[Token], None] | None = None):
        self.env = env
        self.tokens_per_request = tokens_per_request
        self._record_completion = record_completion_cb
        self._request_complete_cb = request_complete_cb
        self.max_active_requests = max_active_requests
        self.attn_dp_group_size = max(1, int(attn_dp_group_size))
        self._defrag_v0_debug_state = defrag_v0_debug_state
        self._token_begin_cb = token_begin_cb
        self.first_attention: AttentionWorker | None = None
        self._next_tid = 0
        self._request_state: Dict[int, int] = {}
        self._request_start_time: Dict[int, float] = {}
        self._request_home_gpu: Dict[int, int] = {}
        self._next_home_gpu = 0
        self._pending_queue = deque()

    def set_first_attention(self, attention_worker: AttentionWorker):
        self.first_attention = attention_worker

    @property
    def active_requests(self) -> int:
        return len(self._request_state)

    @property
    def pending_queue_len(self) -> int:
        return len(self._pending_queue)

    def admit_request(self, request_id: int, arrival_time: float):
        if self.max_active_requests is None or self.active_requests < self.max_active_requests:
            self._start_request_now(request_id, arrival_time)
        else:
            self._pending_queue.append((request_id, arrival_time))

    def _start_request_now(self, request_id: int, arrival_time: float):
        if request_id in self._request_state:
            raise ValueError(f"Request {request_id} already started")
        self._request_state[request_id] = 0
        self._request_start_time[request_id] = arrival_time
        self._request_home_gpu[request_id] = (self._next_home_gpu % self.attn_dp_group_size)
        self._next_home_gpu += 1

        dbg = self._defrag_v0_debug_state
        if (
            dbg is not None
            and dbg.enabled
            and not dbg.triggered
            and self.max_active_requests is not None
            and self.active_requests >= self.max_active_requests
        ):
            dbg.triggered = True
            dbg.trigger_time = self.env.now
            dbg.trigger_request_id = request_id
            dbg.trigger_active_requests = self.active_requests
            dbg.trigger_pending_queue_len = self.pending_queue_len

        self._dispatch_next_token(request_id)

    def handle_token_completion(self, token: Token, completion_time: float):
        self._record_completion(token, completion_time)
        rid = token.request_id
        if rid not in self._request_state:
            raise RuntimeError(f"Completion received for unknown request {rid}")

        if token.token_index + 1 < self.tokens_per_request:
            self._dispatch_next_token(rid)
        else:
            # Request fully finished; drop state for bookkeeping
            self._request_state.pop(rid, None)
            start_time = self._request_start_time.pop(rid, token.birth_time)
            self._request_home_gpu.pop(rid, None)
            if self._request_complete_cb is not None:
                self._request_complete_cb(rid, completion_time - start_time)
            self._maybe_admit_queued_requests()

    def _dispatch_next_token(self, request_id: int):
        if self.first_attention is None:
            raise RuntimeError("First attention worker not initialized yet.")

        next_idx = self._request_state.get(request_id)
        if next_idx is None:
            raise RuntimeError(f"Request {request_id} not tracked for dispatch.")
        if next_idx >= self.tokens_per_request:
            return

        home_gpu = self._request_home_gpu.get(request_id)
        if home_gpu is None:
            raise RuntimeError(f"Request {request_id} missing home_gpu assignment.")

        token = Token(
            tid=self._next_tid,
            birth_time=self.env.now,
            request_id=request_id,
            token_index=next_idx,
            home_gpu=int(home_gpu),
        )
        self._next_tid += 1
        self._request_state[request_id] = next_idx + 1
        if self._token_begin_cb is not None:
            self._token_begin_cb(token)
        self.first_attention.enqueue(token)

    def _maybe_admit_queued_requests(self):
        if self.max_active_requests is None:
            while self._pending_queue:
                rid, ts = self._pending_queue.popleft()
                self._start_request_now(rid, ts)
            return

        while self._pending_queue and self.active_requests < self.max_active_requests:
            rid, ts = self._pending_queue.popleft()
            self._start_request_now(rid, ts)


class FinalCompletionTracker:
    """
    Aggregates completions for the final layer so a token only finishes after
    every routed expert replica returned.
    """

    def __init__(self, final_layer_idx: int, completion_callback):
        self.final_layer_idx = final_layer_idx
        self.completion_callback = completion_callback
        self._pending_counts = defaultdict(int)

    def record_completion(self, token: Token, now: float):
        expected = token.layer_fanout.get(self.final_layer_idx)
        if expected is None or expected <= 0:
            raise RuntimeError(
                f"No fanout recorded for token {token.tid} at final layer {self.final_layer_idx}"
            )

        key = token.tid
        new_count = self._pending_counts[key] + 1
        if new_count < expected:
            self._pending_counts[key] = new_count
            return
        if new_count == expected:
            self._pending_counts.pop(key, None)
            self.completion_callback(token, now)
            return
        raise RuntimeError(
            f"Token {token.tid} received more final completions ({new_count}) than expected ({expected})"
        )


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


# ------------------------------
# Expert worker
# ------------------------------

class ExpertWorker:
    """
    Represents one expert worker that is shared across all layers.
    It owns `num_queues_per_worker_per_layer` queues per layer (for experts assigned
    to this worker), i.e. a total of `num_layers * num_queues_per_worker_per_layer`
    queues.

    It always picks the *longest* queue ACROSS ALL LAYERS to form the largest batch,
    up to MAX_BATCH_SIZE.
    Batch compute time depends on the formed batch size, using a loaded profile.
    Then each token in the batch is sent to the NEXT layer's attention worker
    (or marked finished if this is the last layer).
    """
    def __init__(
        self,
        env,
        worker_idx,
        num_layers,
        num_queues_per_worker_per_layer,
        max_batch_size,
        compute_time_lookup,
        net_t_expert_to_attn,
        attention_layers,
        final_completion_tracker: FinalCompletionTracker,
        *,
        net_delay_fn: Callable[[int, int, int], float] | None = None,
        ingress_ports: List[IngressPort] | None = None,
        batch_stats=None,
        defrag_v0_debug_state: DefragV0DebugState | None = None,
        scheduling_policy: str = "longest_queue_first",
        routing_top_k: int = ROUTING_TOP_K,
    ):
        self.env = env
        self.worker_idx = worker_idx
        self.num_layers = num_layers
        self.num_queues_per_worker_per_layer = num_queues_per_worker_per_layer
        self.max_batch_size = max_batch_size
        self._compute_time_lookup = compute_time_lookup
        self.net_t_expert_to_attn = net_t_expert_to_attn
        self.net_delay_fn = net_delay_fn
        self.ingress_ports = ingress_ports
        self.attention_layers = attention_layers
        self.final_completion_tracker = final_completion_tracker
        self._batch_stats = batch_stats
        self._total_queue_length = 0
        self.scheduling_policy = scheduling_policy
        self._defrag_v0_debug_state = defrag_v0_debug_state
        self._inflight_batch_size = 0
        self._inflight_attention = False
        self.routing_top_k = max(1, int(routing_top_k))

        # Per-GPU attention queue. This models the constraint that a GPU cannot
        # compute attention while it is computing an expert batch (and vice
        # versa). Tokens are enqueued here by AttentionWorker.enqueue() based on
        # token.home_gpu.
        self.attention_queue: deque[tuple["AttentionWorker", Token]] = deque()

        # Per-layer, per-expert queues:
        # queues[layer_idx][local_queue_idx]
        self.queues = [
            [deque() for _ in range(num_queues_per_worker_per_layer)]
            for _ in range(num_layers)
        ]

        # Event used for waking the worker when new work arrives
        self.has_work = env.event()

        # Start main process
        self.proc = env.process(self.run())

    @property
    def total_queue_length(self):
        return self._total_queue_length

    @property
    def attention_queue_length(self) -> int:
        return len(self.attention_queue)

    @property
    def inflight_attention(self) -> bool:
        return bool(self._inflight_attention)

    def enqueue(self, layer_idx: int, local_queue_idx: int, token: Token):
        q = self.queues[layer_idx][local_queue_idx]
        was_empty = (self._total_queue_length == 0 and not self.attention_queue)
        q.append(token)
        self._total_queue_length += 1
        # Wake the worker if it was idle
        if was_empty and not self.has_work.triggered:
            self.has_work.succeed()

    def enqueue_attention(self, attention_layer: "AttentionWorker", token: Token):
        was_empty = (self._total_queue_length == 0 and not self.attention_queue)
        self.attention_queue.append((attention_layer, token))
        if was_empty and not self.has_work.triggered:
            self.has_work.succeed()

    def run(self):
        """
        Main loop for the expert worker.

        Scheduling is at LAYER granularity, matching the real C++ system:
        - UnifiedDefraggingLayerScheduler picks a layer_id
        - get_batch_from_layer() merges ALL tokens for that layer
        - Expert executor uses grouped GEMM to process all experts together

        The scheduling policy controls HOW the layer is selected:
          - "longest_queue_first": pick the layer with the most total tokens.
          - "defragging_v0": use lookahead scoring matching C++.
        """
        while True:
            if self._total_queue_length == 0 and not self.attention_queue:
                # No work: go to sleep
                self.has_work = self.env.event()
                yield self.has_work

            task_kind, payload = self._select_next_task()
            if task_kind == "attention":
                attention_layer, token = payload
                self._inflight_attention = True
                yield self.env.timeout(attention_layer.attn_service_t)
                self._inflight_attention = False
                # Schedule routing + dispatch asynchronously so this GPU can
                # start the next compute immediately.
                self.env.process(attention_layer.route_and_dispatch(token))
                continue

            # payload is layer_idx (or None if no work)
            chosen_layer_idx = payload
            if chosen_layer_idx is None:
                continue

            # Form a batch from ALL queues for this layer (matching TokenBatch::merge)
            batch: List[Token] = []
            for local_queue in self.queues[chosen_layer_idx]:
                while local_queue and len(batch) < self.max_batch_size:
                    batch.append(local_queue.popleft())
                    self._total_queue_length -= 1

            if not batch:
                continue

            # Track batch size statistics if requested.
            if self._batch_stats is not None:
                self._batch_stats["total_batch_size"] += len(batch)
                self._batch_stats["total_batches"] += 1

            # Expert compute with grouped GEMM speedup
            # (matches sync/tbo which apply GROUP_GEMM_SPEEDUP_FACTOR)
            self._inflight_batch_size = len(batch)
            base_compute_t = self._compute_time_lookup[len(batch)]
            compute_t = base_compute_t * GROUP_GEMM_SPEEDUP_FACTOR
            yield self.env.timeout(compute_t)

            # Network delay to attention / completion.
            # With congestion-aware simulation, each destination GPU's ingress
            # port fair-shares bandwidth among concurrent senders. We group
            # tokens by home_gpu and fire concurrent transfers.
            if self.ingress_ports is not None and self.net_delay_fn is not None:
                # Group tokens by destination (home_gpu)
                tokens_per_dst: Dict[int, int] = defaultdict(int)
                for token in batch:
                    tokens_per_dst[int(token.home_gpu)] += 1

                # Fire concurrent transfers to each destination's ingress port
                transfer_events = []
                for dst_gpu, count in tokens_per_dst.items():
                    solo_delay = float(
                        self.net_delay_fn(int(self.worker_idx), dst_gpu, count)
                    )
                    if solo_delay > 0:
                        transfer_events.append(
                            self.ingress_ports[dst_gpu].transfer(int(self.worker_idx), solo_delay)
                        )

                if transfer_events:
                    yield simpy.events.AllOf(self.env, transfer_events)
            else:
                delay = self.net_t_expert_to_attn
                yield self.env.timeout(delay)
            self._inflight_batch_size = 0

            # Route tokens onward
            for token in batch:
                if chosen_layer_idx == self.num_layers - 1:
                    # Last layer expert => feed into final completion tracker
                    self.final_completion_tracker.record_completion(token, self.env.now)
                else:
                    # Send to next layer's attention worker pending list
                    next_attn = self.attention_layers[chosen_layer_idx + 1]
                    next_attn.notify_expert_completion(token)

    def _select_next_task(self):
        """
        Select the next unit of work for this GPU:
          - an expert batch from a layer (all queues for that layer), or
          - one attention token from this GPU's attention queue.

        Returns:
          ("attention", (attention_layer, token)) or
          ("expert", layer_idx_or_None)
        """
        attention_score = (
            float(len(self.attention_queue)) * float(self.routing_top_k)
            if self.attention_queue
            else 0.0
        )

        # Use layer-level scheduling (matching real system behavior)
        if self.scheduling_policy == "defragging_v0":
            layer_idx, score = self._select_next_layer_defragging()
        else:
            layer_idx, score = self._select_next_layer_longest_first()

        if self.attention_queue and (layer_idx is None or attention_score > float(score)):
            return "attention", self.attention_queue.popleft()
        return "expert", layer_idx

    def _select_next_queue_longest_queue_first(self):
        """Original policy: pick the single longest queue across all layers."""
        max_q = None
        max_len = 0
        chosen_layer_idx = None

        for layer_idx, layer_queues in enumerate(self.queues):
            for q in layer_queues:
                q_len = len(q)
                if q_len > max_len:
                    max_len = q_len
                    max_q = q
                    chosen_layer_idx = layer_idx

        if max_q is None or max_len == 0:
            return None, None, 0.0
        return chosen_layer_idx, max_q, float(max_len)

    def _select_next_layer_longest_first(self) -> tuple[int | None, float]:
        """
        Layer-level scheduling: pick the layer with the most total tokens.
        This matches the real system behavior where scheduling is at layer
        granularity, not per-expert queue.
        """
        best_layer_idx = None
        best_score = 0.0

        for layer_idx, layer_queues in enumerate(self.queues):
            layer_total = sum(len(q) for q in layer_queues)
            if layer_total > best_score:
                best_score = float(layer_total)
                best_layer_idx = layer_idx

        return best_layer_idx, best_score

    def _select_next_layer_defragging(self) -> tuple[int | None, float]:
        """
        Layer-level defragging scheduler matching C++ UnifiedDefraggingLayerScheduler.

        Computes a score for each layer based on:
        1. Immediate tokens in the layer (sum across all local expert queues)
        2. Lookahead score: decayed sum of tokens in future layers

        This matches the real system where scheduling is at layer granularity,
        then all tokens for that layer are batched together for grouped GEMM.
        """
        n_layers = self.num_layers
        if n_layers <= 0:
            return None, 0.0

        # Scheduler parameters (match C++ defaults)
        weight_decay = 0.8
        lookahead_steps = 4

        # Compute total tokens per layer
        queues = self.queues
        layer_totals = [0] * n_layers
        for layer_idx in range(n_layers):
            layer_totals[layer_idx] = sum(len(q) for q in queues[layer_idx])

        # Early exit if no work
        total_tokens = sum(layer_totals)
        if total_tokens == 0:
            return None, 0.0

        # Compute per-layer scores with lookahead
        scores = [0.0] * n_layers
        for i in range(n_layers):
            # Immediate tokens in this layer
            immediate = float(layer_totals[i])
            if immediate <= 0:
                continue

            # Compute lookahead score
            lookahead_score = 0.0
            decay = weight_decay
            for k in range(1, lookahead_steps):
                cur_layer = (i + k) % n_layers
                num_tokens_cur_layer = layer_totals[cur_layer]
                if num_tokens_cur_layer > 0:
                    lookahead_score += num_tokens_cur_layer * decay
                decay *= weight_decay

            scores[i] = immediate + lookahead_score

        # Find best layer
        best_layer_idx = None
        best_score = 0.0
        for i in range(n_layers):
            if scores[i] > best_score:
                best_score = scores[i]
                best_layer_idx = i

        return best_layer_idx, best_score

    def _select_next_queue_defragging_v0(self):
        """
        "Defragging" policy inspired by GroupLayerScheduler::schedule.

        We treat each (layer_idx, local_expert_idx) as a "layer_group". For a
        given starting layer i and local expert j, we look ahead across layers
        for the same local expert j and accumulate a decayed score based on how
        many tokens are queued in those future layers. Queues that are busy in
        both the current and upcoming layers are preferred.
        """
        n_layers = self.num_layers
        n_groups = self.num_queues_per_worker_per_layer
        if n_layers <= 0 or n_groups <= 0:
            return None, None

        # scheduler parameters
        weight_decay = 0.95
        lookahead_steps = 4

        best_score = 0.0
        best_layer_idx = None
        best_group_idx = None

        # Precompute per-layer total queue lengths for this worker so that
        # we can approximate the "lookahead" load across all experts in a
        # given layer.
        queues = self.queues
        layer_totals = [0] * n_layers
        queue_len_matrix = [[0] * n_groups for _ in range(n_layers)]
        for layer_idx in range(n_layers):
            total = 0
            for group_idx, q in enumerate(queues[layer_idx]):
                q_len = len(q)
                queue_len_matrix[layer_idx][group_idx] = q_len
                total += q_len
            layer_totals[layer_idx] = total

        inv_n_groups = 1.0 / float(n_groups)
        lookahead_scores_by_layer = [0.0] * n_layers
        score_matrix: List[List[float | None]] = [
            [None] * n_groups for _ in range(n_layers)
        ]

        for i in range(n_layers):
            # Compute lookahead score shared by all groups at layer i.
            lookahead_score = 0.0
            decay = weight_decay
            for k in range(1, lookahead_steps):
                cur_layer = (i + k) % n_layers
                num_tokens_cur_layer = layer_totals[cur_layer]
                if num_tokens_cur_layer > 0:
                    lookahead_score += num_tokens_cur_layer * decay * inv_n_groups
                decay *= weight_decay
            lookahead_scores_by_layer[i] = lookahead_score

            # Now compute per-(layer, group) scores, only for non-empty queues
            # at the current layer. Queues with more tokens in the current
            # layer and busy future layers will have higher scores.
            layer_queues = queues[i]
            for j in range(n_groups):
                q = layer_queues[j]
                local_len = len(q)
                if local_len <= 0:
                    continue
                score = lookahead_score + float(local_len)
                score_matrix[i][j] = score
                if best_layer_idx is None or score > best_score:
                    best_score = score
                    best_layer_idx = i
                    best_group_idx = j

        dbg = self._defrag_v0_debug_state
        if (
            dbg is not None
            and dbg.enabled
            and dbg.triggered
        ):
            active_requests_at_log_time = (
                dbg.get_active_requests() if dbg.get_active_requests is not None else None
            )
            pending_queue_len_at_log_time = (
                dbg.get_pending_queue_len() if dbg.get_pending_queue_len is not None else None
            )
            global_expert_queued_tokens = (
                dbg.get_global_expert_queued_tokens()
                if dbg.get_global_expert_queued_tokens is not None
                else None
            )
            global_expert_inflight_tokens = (
                dbg.get_global_expert_inflight_tokens()
                if dbg.get_global_expert_inflight_tokens is not None
                else None
            )
            per_worker_expert_queued_tokens = (
                dbg.get_per_worker_expert_queued_tokens()
                if dbg.get_per_worker_expert_queued_tokens is not None
                else None
            )
            per_worker_expert_inflight_tokens = (
                dbg.get_per_worker_expert_inflight_tokens()
                if dbg.get_per_worker_expert_inflight_tokens is not None
                else None
            )
            attention_users = (
                dbg.get_attention_users() if dbg.get_attention_users is not None else None
            )
            attention_queue = (
                dbg.get_attention_queue() if dbg.get_attention_queue is not None else None
            )
            pending_tokens_total = (
                dbg.get_pending_tokens_total()
                if dbg.get_pending_tokens_total is not None
                else None
            )
            pending_tokens_per_layer = (
                dbg.get_pending_tokens_per_layer()
                if dbg.get_pending_tokens_per_layer is not None
                else None
            )
            tokens_created = (
                dbg.get_tokens_created() if dbg.get_tokens_created is not None else None
            )
            tokens_completed = (
                dbg.get_tokens_completed() if dbg.get_tokens_completed is not None else None
            )
            tokens_inflight_unique = (
                (int(tokens_created) - int(tokens_completed))
                if tokens_created is not None and tokens_completed is not None
                else None
            )

            dbg.logged = True
            dbg.log_time = self.env.now
            dbg.worker_idx = self.worker_idx
            dbg.entry = {
                "scheduling_policy": "defragging_v0",
                "worker_idx": self.worker_idx,
                "log_time": self.env.now,
                "trigger_time": dbg.trigger_time,
                "trigger_request_id": dbg.trigger_request_id,
                "trigger_active_requests": dbg.trigger_active_requests,
                "trigger_pending_queue_len": dbg.trigger_pending_queue_len,
                "active_requests_at_log_time": active_requests_at_log_time,
                "pending_queue_len_at_log_time": pending_queue_len_at_log_time,
                "global_expert_queued_tokens": global_expert_queued_tokens,
                "global_expert_inflight_tokens": global_expert_inflight_tokens,
                "global_expert_total_tokens": (
                    (int(global_expert_queued_tokens) + int(global_expert_inflight_tokens))
                    if global_expert_queued_tokens is not None and global_expert_inflight_tokens is not None
                    else None
                ),
                "per_worker_expert_queued_tokens": per_worker_expert_queued_tokens,
                "per_worker_expert_inflight_tokens": per_worker_expert_inflight_tokens,
                "attention_users": attention_users,
                "attention_queue": attention_queue,
                "pending_tokens_total": pending_tokens_total,
                "pending_tokens_per_layer": pending_tokens_per_layer,
                "tokens_created": tokens_created,
                "tokens_completed": tokens_completed,
                "tokens_inflight_unique": tokens_inflight_unique,
                "weight_decay": weight_decay,
                "lookahead_steps": lookahead_steps,
                "queue_len_matrix": queue_len_matrix,
                "layer_totals": layer_totals,
                "lookahead_scores_by_layer": lookahead_scores_by_layer,
                "score_matrix": score_matrix,
                "best_layer_idx": best_layer_idx,
                "best_group_idx": best_group_idx,
                "best_score": best_score,
            }

        if best_layer_idx is None or best_group_idx is None:
            return None, None, 0.0
        return best_layer_idx, queues[best_layer_idx][best_group_idx], float(best_score)


# ------------------------------
# Attention + gating worker
# ------------------------------

class AttentionWorker:
    """
    Maintains a pending list of partial expert completions for this layer and
    hands fully-ready tokens to a *per-GPU* attention queue. Each token's
    `home_gpu` selects which GPU processes its attention compute, and that GPU
    serializes attention with expert compute.

    Semantics:
      - Tokens "arrive" to the attention system either from the request source
        (layer 0) or when all top-k experts in the previous layer finish.
      - Attention is queued per GPU (token.home_gpu). Across all GPUs, at most
        one attention token per GPU can be processed at a time.
      - After attention/gating:
          * the token is routed to all top-k experts from the profile-driven
            router for this layer;
          * a fixed NET_T_ATTN_TO_EXPERT delay is paid before enqueuing the
            token into every selected expert worker for this layer.
    """
    def __init__(self, env, layer_idx,
                 total_expert_count,
                 ep_group_size,
                 attn_service_t,
                 net_t_attn_to_expert,
                 expert_workers,
                 profile_router: ProfileDrivenRouter,
                 *,
                 net_delay_fn: Callable[[int, int, int], float] | None = None,
                 ingress_ports: List[IngressPort] | None = None):
        self.env = env
        self.layer_idx = layer_idx
        self.total_expert_count = total_expert_count
        self.ep_group_size = ep_group_size
        self.attn_service_t = attn_service_t
        self.net_t_attn_to_expert = net_t_attn_to_expert
        self.net_delay_fn = net_delay_fn
        self.ingress_ports = ingress_ports
        self.expert_workers = expert_workers
        self.profile_router = profile_router

        self.pending_tokens: Dict[int, Token] = {}
        self.pending_counts = defaultdict(int)

        # Derived:
        assert total_expert_count % ep_group_size == 0, \
            "total_expert_count must be divisible by ep_group_size"
        self.queues_per_worker = total_expert_count // ep_group_size

        self._router_device = torch.device("cpu")
        self._router_dtype = torch.float32

    def enqueue(self, token: Token):
        """
        Schedule attention + routing for a single token by enqueuing it into
        the per-GPU attention queue for the token's home GPU.
        """
        home_gpu = int(token.home_gpu)
        if home_gpu < 0 or home_gpu >= len(self.expert_workers):
            raise RuntimeError(
                f"Token {token.tid} has home_gpu={home_gpu} but there are "
                f"{len(self.expert_workers)} GPU workers."
            )
        self.expert_workers[home_gpu].enqueue_attention(self, token)

    def notify_expert_completion(self, token: Token):
        """
        Called by the previous layer's experts as each replica finishes.
        We release the token to the ready queue only after every replica returns.
        """
        if self.layer_idx == 0:
            raise RuntimeError("Layer 0 attention should not receive expert completions.")

        prev_layer = self.layer_idx - 1
        expected = token.layer_fanout.get(prev_layer)
        if expected is None or expected <= 0:
            raise RuntimeError(
                f"Token {token.tid} missing fanout metadata for layer {prev_layer}"
            )

        key = token.tid
        current = self.pending_counts[key] + 1
        if current < expected:
            self.pending_counts[key] = current
            self.pending_tokens.setdefault(key, token)
            return

        if current == expected:
            self.pending_counts.pop(key, None)
            pending_token = self.pending_tokens.pop(key, token)
            self.enqueue(pending_token)
            return

        raise RuntimeError(
            f"Token {token.tid} received {current} completions at layer {self.layer_idx} "
            f"but only {expected} were expected."
        )

    def route_and_dispatch(self, token: Token):
        """
        Routing + network delay + expert enqueue after attention compute.

        Attention compute time itself is modeled by the GPU worker that popped
        this token from its per-GPU attention queue. This method is scheduled
        asynchronously so GPU compute can overlap with network transfer.
        """
        # Decide routing for all top-k experts
        global_expert_ids = self._route_token(token)
        fanout = len(global_expert_ids)
        if fanout == 0:
            raise RuntimeError(f"Router returned no experts for token {token.tid}")
        token.layer_fanout[self.layer_idx] = fanout

        routes: List[tuple[int, int]] = []
        dest_workers: set[int] = set()
        for global_expert_id in global_expert_ids:
            worker_idx = global_expert_id // self.queues_per_worker
            local_queue_idx = global_expert_id % self.queues_per_worker
            routes.append((worker_idx, local_queue_idx))
            dest_workers.add(worker_idx)

        # Network delay to expert queues. With congestion-aware simulation,
        # each destination worker's ingress port fair-shares bandwidth among
        # concurrent senders. We fire one transfer per unique destination
        # worker and wait for all to complete concurrently.
        if self.ingress_ports is not None and self.net_delay_fn is not None:
            transfer_events = []
            for worker_idx in dest_workers:
                solo_delay = float(
                    self.net_delay_fn(int(token.home_gpu), int(worker_idx), 1)
                )
                if solo_delay > 0:
                    transfer_events.append(
                        self.ingress_ports[worker_idx].transfer(int(token.home_gpu), solo_delay)
                    )

            if transfer_events:
                yield simpy.events.AllOf(self.env, transfer_events)
        else:
            delay = self.net_t_attn_to_expert
            yield self.env.timeout(delay)

        # Send token copies into each chosen expert queue in THIS layer
        for worker_idx, local_queue_idx in routes:
            expert_worker = self.expert_workers[worker_idx]
            expert_worker.enqueue(self.layer_idx, local_queue_idx, token)

    def _route_token(self, token: Token) -> List[int]:
        """Return the list of expert ids selected by the profile-driven router."""
        token_indices = torch.tensor(
            [token.token_index],
            device=self._router_device,
            dtype=torch.int64,
        )
        _, topk_ids = self.profile_router.route(
            request_ids=[token.request_id],
            token_indices=token_indices,
            layer_id=self.layer_idx,
            top_k=self.profile_router.top_k,
            device=self._router_device,
            dtype=self._router_dtype,
        )
        topk_ids_cpu = topk_ids[0].to("cpu").tolist()
        selected = [int(expert_id) for expert_id in topk_ids_cpu if expert_id >= 0]
        if not selected:
            raise RuntimeError(
                f"Profile returned only placeholder expert ids for token {token.tid} at layer {self.layer_idx}"
            )
        return selected


# ------------------------------
# Request source
# ------------------------------

def request_source(env,
                   total_requests: int,
                   arrival_rate: float,
                   request_manager: RequestManager):
    """
    Poisson arrival process with rate `arrival_rate` over requests.
    Each arriving request immediately dispatches token 0; subsequent tokens
    are sent only after the previous token completes.
    """
    for rid in range(total_requests):
        request_manager.admit_request(rid, arrival_time=env.now)

        if rid < total_requests - 1:
            # Poisson arrivals => Exp(lambda) inter-arrival
            inter_arrival = random.expovariate(arrival_rate)
            yield env.timeout(inter_arrival)


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
    expert_profile_path=EXPERT_COMPUTE_PROFILE_PATH,
    profile_routing_path=PROFILE_ROUTING_PATH,
    routing_top_k=ROUTING_TOP_K,
    global_request_max_batch_size=GLOBAL_REQUEST_MAX_BATCH_SIZE,
    attn_dp_group_size=ATTN_DP_GROUP_SIZE,
    tokens_after_reaching_max_bs: int | None = None,
    enable_defrag_v0_debug_log: bool = False,
    scheduling_policy: str = SCHEDULING_POLICY,
    per_token_begin_cb: Callable[[Token], None] | None = None,
    per_token_stats_cb: Callable[[Token, float], None] | None = None,
    net_delay_fn: Callable[[int, int, int], float] | None = None,
    n_gpu_per_host: int | None = None,
):
    random.seed(RNG_SEED)
    env = simpy.Environment()
    expected_total_tokens = total_requests * tokens_per_request
    progress_tracker = ProgressTracker(expected_total_tokens)

    # For metrics
    completion_times = {}  # tid -> absolute completion time
    token_latencies = {}   # tid -> completion_time - birth_time
    request_latency_values: List[float] = []

    if per_token_stats_cb is None:
        def record_completion(token: Token, t_complete: float):
            completion_times[token.tid] = t_complete
            token_latencies[token.tid] = t_complete - token.birth_time
            progress_tracker.update(len(completion_times))
    else:
        def record_completion(token: Token, t_complete: float):
            completion_times[token.tid] = t_complete
            token_latencies[token.tid] = t_complete - token.birth_time
            progress_tracker.update(len(completion_times))
            if token.sampled_for_stats:
                per_token_stats_cb(token, t_complete)

    def record_request_completion(request_id: int, latency: float):
        request_latency_values.append(latency)

    defrag_v0_debug_state = (
        DefragV0DebugState(enabled=True)
        if enable_defrag_v0_debug_log
        else None
    )

    # Attention parallelism: ATTN_DP_GROUP_SIZE workers globally. Caller is
    # expected to set this explicitly; we still clamp it to a positive int.
    attn_dp_group_size = max(1, int(attn_dp_group_size))
    if int(attn_dp_group_size) != int(ep_group_size):
        raise ValueError(
            "sim_async models attention and expert compute sharing the same GPUs; "
            "set attn_dp_group_size == ep_group_size to enforce mutual exclusion."
        )

    request_manager = RequestManager(
        env=env,
        tokens_per_request=tokens_per_request,
        record_completion_cb=record_completion,
        request_complete_cb=record_request_completion,
        max_active_requests=global_request_max_batch_size,
        attn_dp_group_size=attn_dp_group_size,
        defrag_v0_debug_state=defrag_v0_debug_state,
        token_begin_cb=per_token_begin_cb,
    )
    if defrag_v0_debug_state is not None:
        defrag_v0_debug_state.get_active_requests = lambda: request_manager.active_requests
        defrag_v0_debug_state.get_pending_queue_len = lambda: request_manager.pending_queue_len
        defrag_v0_debug_state.get_tokens_created = lambda: request_manager._next_tid
        defrag_v0_debug_state.get_tokens_completed = lambda: len(completion_times)

    final_completion_tracker = FinalCompletionTracker(
        final_layer_idx=num_layers - 1,
        completion_callback=request_manager.handle_token_completion,
    )

    # Build expert workers (shared across all layers) and per-layer attention workers.
    expert_workers = []
    attention_layers = []

    queues_per_worker_per_layer = total_expert_count // ep_group_size

    # Aggregate statistics for per-expert batch sizes.
    batch_stats = {
        "total_batch_size": 0.0,
        "total_batches": 0,
    }

    # Load compute profile into a lookup table and validate coverage
    compute_time_lookup = expert_compute_time_lookup_table_from_profile(
        expert_profile_path,
        max_batch_size,
        ticks_per_millisecond=TICKS_PER_MILLISECOND,
    )

    profile_router = build_profile_router(
        profile_path=profile_routing_path,
        num_experts=total_expert_count,
        top_k=routing_top_k,
    )

    # Build experts first with attention pointer set to None.
    # Then create attention workers and patch each expert's attention_layers reference.

    # Create one IngressPort per GPU for congestion-aware network simulation.
    # Each port has two independent FIFO channels (NVLink and RDMA).
    # Transfers are classified by topology and queued on the appropriate
    # channel; each channel processes one transfer at a time at full
    # bandwidth.  The port is shared by both dispatch (attn→expert) and
    # return (expert→attn) traffic targeting this GPU.
    ingress_ports: List[IngressPort] | None = None
    if net_delay_fn is not None:
        _n_gpu_per_host = n_gpu_per_host if n_gpu_per_host is not None else ep_group_size
        ingress_ports = [
            IngressPort(env, gpu_idx=i, n_gpu_per_host=_n_gpu_per_host)
            for i in range(ep_group_size)
        ]

    # Create expert workers (global across all layers)
    for w in range(ep_group_size):
        worker = ExpertWorker(
            env=env,
            worker_idx=w,
            num_layers=num_layers,
            num_queues_per_worker_per_layer=queues_per_worker_per_layer,
            max_batch_size=max_batch_size,
            compute_time_lookup=compute_time_lookup,
            net_t_expert_to_attn=NET_T_EXPERT_TO_ATTN,
            net_delay_fn=net_delay_fn,
            ingress_ports=ingress_ports,
            attention_layers=None,  # temp, will fix after we create them
            final_completion_tracker=final_completion_tracker,
            batch_stats=batch_stats,
            defrag_v0_debug_state=defrag_v0_debug_state,
            scheduling_policy=scheduling_policy,
            routing_top_k=routing_top_k,
        )
        expert_workers.append(worker)

    if defrag_v0_debug_state is not None:
        defrag_v0_debug_state.get_attention_users = lambda: sum(
            1 for worker in expert_workers if worker.inflight_attention
        )
        defrag_v0_debug_state.get_attention_queue = lambda: sum(
            int(worker.attention_queue_length) for worker in expert_workers
        )

    if defrag_v0_debug_state is not None:
        def _global_queued_tokens() -> int:
            total = 0
            for worker in expert_workers:
                for layer_queues in worker.queues:
                    for q in layer_queues:
                        total += len(q)
            return total

        def _global_inflight_tokens() -> int:
            return sum(int(w._inflight_batch_size) for w in expert_workers)

        def _per_worker_queued_tokens() -> List[int]:
            totals: List[int] = []
            for worker in expert_workers:
                total = 0
                for layer_queues in worker.queues:
                    for q in layer_queues:
                        total += len(q)
                totals.append(total)
            return totals

        def _per_worker_inflight_tokens() -> List[int]:
            return [int(w._inflight_batch_size) for w in expert_workers]

        defrag_v0_debug_state.get_global_expert_queued_tokens = _global_queued_tokens
        defrag_v0_debug_state.get_global_expert_inflight_tokens = _global_inflight_tokens
        defrag_v0_debug_state.get_per_worker_expert_queued_tokens = _per_worker_queued_tokens
        defrag_v0_debug_state.get_per_worker_expert_inflight_tokens = _per_worker_inflight_tokens

    # Now create attention workers (they need expert_layers)
    for layer in range(num_layers):
        attn = AttentionWorker(
            env=env,
            layer_idx=layer,
            total_expert_count=total_expert_count,
            ep_group_size=ep_group_size,
            attn_service_t=ATTN_SERVICE_T,
            net_t_attn_to_expert=NET_T_ATTN_TO_EXPERT,
            net_delay_fn=net_delay_fn,
            ingress_ports=ingress_ports,
            expert_workers=expert_workers,
            profile_router=profile_router,
        )
        attention_layers.append(attn)

    request_manager.set_first_attention(attention_layers[0])
    if defrag_v0_debug_state is not None:
        def _pending_tokens_total() -> int:
            return sum(len(attn.pending_tokens) for attn in attention_layers)

        def _pending_tokens_per_layer() -> List[int]:
            return [len(attn.pending_tokens) for attn in attention_layers]

        defrag_v0_debug_state.get_pending_tokens_total = _pending_tokens_total
        defrag_v0_debug_state.get_pending_tokens_per_layer = _pending_tokens_per_layer

    # Now that we have attention_layers, fix the forward pointers in experts
    for w in expert_workers:
        w.attention_layers = attention_layers

    # Create the request source
    env.process(request_source(
        env,
        total_requests=total_requests,
        arrival_rate=arrival_rate,
        request_manager=request_manager,
    ))

    steps_executed = 0
    saturation_step: int | None = None
    saturation_tokens_completed: int | None = None
    saturation_time: float | None = None
    if tokens_after_reaching_max_bs is not None:
        tokens_after_reaching_max_bs = int(tokens_after_reaching_max_bs)
        if tokens_after_reaching_max_bs < 0:
            raise ValueError("tokens_after_reaching_max_bs must be >= 0 or None")

        # Step event-by-event so we can stop shortly after the system reaches
        # the configured max active request batch size (i.e., saturation), once
        # enough additional tokens have completed.
        try:
            while True:
                env.step()
                steps_executed += 1

                if (
                    saturation_step is None
                    and request_manager.max_active_requests is not None
                    and request_manager.active_requests
                    >= request_manager.max_active_requests
                ):
                    saturation_step = steps_executed
                    saturation_tokens_completed = len(completion_times)
                    saturation_time = float(env.now)

                if (
                    saturation_tokens_completed is not None
                    and (len(completion_times) - saturation_tokens_completed)
                    >= tokens_after_reaching_max_bs
                ):
                    break
        except simpy.core.EmptySchedule:
            pass
    else:
        # Run until all tokens are completed.
        # Easiest is to run until the event queue drains; since we only
        # generate a finite number of tokens per request and they all eventually
        # complete, the sim will naturally finish.
        env.run()

    if len(completion_times) >= expected_total_tokens:
        progress_tracker.finalize()
    else:
        progress_tracker.update(len(completion_times), force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Metrics
    stopped_early = len(completion_times) != expected_total_tokens
    if stopped_early and tokens_after_reaching_max_bs is None:
        print(
            "WARNING: some tokens did not complete!",
            len(completion_times),
            "/",
            expected_total_tokens,
        )

    latencies = list(token_latencies.values())
    avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
    makespan = float(env.now)

    request_latencies_ms = [
        latency / TICKS_PER_MILLISECOND for latency in request_latency_values
    ]
    avg_request_latency_ms = (
        sum(request_latencies_ms) / len(request_latencies_ms)
        if request_latencies_ms
        else float("nan")
    )
    p90_request_latency_ms = (
        percentile(request_latency_values, 0.90) / TICKS_PER_MILLISECOND
        if request_latency_values
        else float("nan")
    )
    p99_request_latency_ms = (
        percentile(request_latency_values, 0.99) / TICKS_PER_MILLISECOND
        if request_latency_values
        else float("nan")
    )

    makespan_ms = makespan / TICKS_PER_MILLISECOND
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
        tail_seconds = ((makespan - float(saturation_time)) / TICKS_PER_MILLISECOND) / 1000.0
        if tail_seconds > 0:
            tail_throughput_req_per_sec = (tail_tokens / float(tokens_per_request)) / tail_seconds

    # Average per-expert batch size across all expert compute invocations.
    if batch_stats["total_batches"] > 0:
        avg_per_expert_batch_size = (
            batch_stats["total_batch_size"] / batch_stats["total_batches"]
        )
    else:
        avg_per_expert_batch_size = 0.0

    if stopped_early:
        print(
            f"Simulation stopped early at time {makespan:.3f} "
            f"(completed_tokens={completed_tokens}/{expected_total_tokens}, "
            f"steps_executed={steps_executed}, saturation_step={saturation_step})"
        )
    else:
        print(f"Simulation finished at time {makespan:.3f}")

    print(f"Average completion time over {len(latencies)} tokens: {avg_latency:.3f}")
    print(f"Average throughput: {avg_throughput_req_per_sec:.3f} requests/sec")
    if tail_throughput_req_per_sec == tail_throughput_req_per_sec and math.isfinite(tail_throughput_req_per_sec):
        print(
            f"Tail throughput (after saturation, {tokens_after_reaching_max_bs} tokens): "
            f"{tail_throughput_req_per_sec:.3f} requests/sec"
        )
    print(f"Average request latency: {avg_request_latency_ms:.3f} ms")
    print(f"P90 request latency: {p90_request_latency_ms:.3f} ms")
    print(f"P99 request latency: {p99_request_latency_ms:.3f} ms")
    print(f"Average per-expert batch size: {avg_per_expert_batch_size:.3f}")

    return {
        "completion_times": completion_times,
        "avg_latency": avg_latency,
        "makespan": makespan,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "stopped_early": stopped_early,
        "steps_executed": steps_executed,
        "saturation_step": saturation_step,
        "saturation_tokens_completed": saturation_tokens_completed,
        "saturation_time": saturation_time,
        "defrag_v0_debug_triggered": (
            defrag_v0_debug_state.triggered
            if defrag_v0_debug_state is not None
            else False
        ),
        "defrag_v0_debug_trigger_time": (
            defrag_v0_debug_state.trigger_time
            if defrag_v0_debug_state is not None
            else None
        ),
        "defrag_v0_debug_logged": (
            defrag_v0_debug_state.logged
            if defrag_v0_debug_state is not None
            else False
        ),
        "defrag_v0_debug_entry": (
            defrag_v0_debug_state.entry
            if defrag_v0_debug_state is not None
            else None
        ),
    }


if __name__ == "__main__":
    run_simulation()
