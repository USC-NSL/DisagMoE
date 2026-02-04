from __future__ import annotations

import simpy
import random
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import math
import sys
import time
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
                 defrag_v0_debug_state: DefragV0DebugState | None = None,
                 token_begin_cb: Callable[[Token], None] | None = None):
        self.env = env
        self.tokens_per_request = tokens_per_request
        self._record_completion = record_completion_cb
        self._request_complete_cb = request_complete_cb
        self.max_active_requests = max_active_requests
        self._defrag_v0_debug_state = defrag_v0_debug_state
        self._token_begin_cb = token_begin_cb
        self.first_attention: AttentionWorker | None = None
        self._next_tid = 0
        self._request_state: Dict[int, int] = {}
        self._request_start_time: Dict[int, float] = {}
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

        token = Token(
            tid=self._next_tid,
            birth_time=self.env.now,
            request_id=request_id,
            token_index=next_idx,
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
        batch_stats=None,
        defrag_v0_debug_state: DefragV0DebugState | None = None,
        scheduling_policy: str = "longest_queue_first",
    ):
        self.env = env
        self.worker_idx = worker_idx
        self.num_layers = num_layers
        self.num_queues_per_worker_per_layer = num_queues_per_worker_per_layer
        self.max_batch_size = max_batch_size
        self._compute_time_lookup = compute_time_lookup
        self.net_t_expert_to_attn = net_t_expert_to_attn
        self.attention_layers = attention_layers
        self.final_completion_tracker = final_completion_tracker
        self._batch_stats = batch_stats
        self._total_queue_length = 0
        self.scheduling_policy = scheduling_policy
        self._defrag_v0_debug_state = defrag_v0_debug_state
        self._inflight_batch_size = 0

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

    def enqueue(self, layer_idx: int, local_queue_idx: int, token: Token):
        q = self.queues[layer_idx][local_queue_idx]
        was_empty = (self._total_queue_length == 0)
        q.append(token)
        self._total_queue_length += 1
        # Wake the worker if it was idle
        if was_empty and not self.has_work.triggered:
            self.has_work.succeed()

    def run(self):
        """
        Main loop for the expert worker.

        The scheduling of which (layer, local_expert) queue to serve next is
        controlled by `self.scheduling_policy`:
          - "longest_queue_first": pick the single longest queue.
          - "defragging_v0": approximate GroupLayerScheduler::schedule from
            the C++ runtime, where each (layer, local_expert) is a
            "layer_group" and we look ahead across layers for that group.
        """
        while True:
            if self._total_queue_length == 0:
                # No work: go to sleep
                self.has_work = self.env.event()
                yield self.has_work

            chosen_layer_idx, chosen_queue = self._select_next_queue()

            if chosen_queue is None:
                # Shouldn't really happen, but be defensive
                continue

            # Form a batch
            batch = []
            while chosen_queue and len(batch) < self.max_batch_size:
                batch.append(chosen_queue.popleft())
                self._total_queue_length -= 1

            # Track batch size statistics if requested.
            if self._batch_stats is not None:
                self._batch_stats["total_batch_size"] += len(batch)
                self._batch_stats["total_batches"] += 1

            # Expert compute
            self._inflight_batch_size = len(batch)
            compute_t = self._compute_time_lookup[len(batch)]
            yield self.env.timeout(compute_t)

            # Network delay to attention / completion
            yield self.env.timeout(self.net_t_expert_to_attn)
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

    def _select_next_queue(self):
        """
        Select the next (layer, queue) to serve according to the configured
        scheduling policy.

        Returns (layer_idx, deque_or_None).
        """
        if self.scheduling_policy == "defragging_v0":
            return self._select_next_queue_defragging_v0()
        # Default / fallback policy.
        return self._select_next_queue_longest_queue_first()

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
            return None, None
        return chosen_layer_idx, max_q

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
                    lookahead_score += num_tokens_cur_layer * decay # * inv_n_groups
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
            return None, None
        return best_layer_idx, queues[best_layer_idx][best_group_idx]


# ------------------------------
# Attention + gating worker
# ------------------------------

class AttentionWorker:
    """
    Maintains a pending list of partial expert completions for this layer and
    hands fully-ready tokens to a *global* attention queue that is shared
    across all layers.

    Semantics:
      - Tokens "arrive" to the attention system either from the request source
        (layer 0) or when all top-k experts in the previous layer finish.
      - The global attention queue is first-come-first-served across all layers.
      - There are ATTN_DP_GROUP_SIZE attention workers that run in parallel.
        In each ATTN_SERVICE_T interval, at most ATTN_DP_GROUP_SIZE tokens can
        complete attention, regardless of which layer they belong to.
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
                 attention_resource: simpy.Resource):
        self.env = env
        self.layer_idx = layer_idx
        self.total_expert_count = total_expert_count
        self.ep_group_size = ep_group_size
        self.attn_service_t = attn_service_t
        self.net_t_attn_to_expert = net_t_attn_to_expert
        self.expert_workers = expert_workers
        self.profile_router = profile_router
        # Shared across *all* layers to model a global FCFS attention queue
        # with ATTN_DP_GROUP_SIZE parallel workers.
        self.attention_resource = attention_resource

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
        the global FCFS attention queue shared across all layers.
        """
        self.env.process(self._process_token(token))

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

    def _process_token(self, token: Token):
        # First-come-first-served attention/gating with limited parallelism.
        # All layers share the same attention_resource so tokens of different
        # layers compete in a single global queue.
        with self.attention_resource.request() as req:
            # Wait for an attention slot.
            yield req
            # Attention/gating compute for this token.
            yield self.env.timeout(self.attn_service_t)

        # Decide routing for all top-k experts
        global_expert_ids = self._route_token(token)
        fanout = len(global_expert_ids)
        if fanout == 0:
            raise RuntimeError(f"Router returned no experts for token {token.tid}")
        token.layer_fanout[self.layer_idx] = fanout

        # Network delay to expert queues (modeled as a single hop before dispatch)
        yield self.env.timeout(self.net_t_attn_to_expert)

        # Send token copies into each chosen expert queue in THIS layer
        for global_expert_id in global_expert_ids:
            worker_idx = global_expert_id // self.queues_per_worker
            local_queue_idx = global_expert_id % self.queues_per_worker
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

    request_manager = RequestManager(
        env=env,
        tokens_per_request=tokens_per_request,
        record_completion_cb=record_completion,
        request_complete_cb=record_request_completion,
        max_active_requests=global_request_max_batch_size,
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

    # Attention parallelism: ATTN_DP_GROUP_SIZE workers globally. Caller is
    # expected to set this explicitly; we still clamp it to a positive int.
    attn_dp_group_size = max(1, int(attn_dp_group_size))
    attention_resource = simpy.Resource(env, capacity=attn_dp_group_size)
    if defrag_v0_debug_state is not None:
        defrag_v0_debug_state.get_attention_users = lambda: len(attention_resource.users)
        defrag_v0_debug_state.get_attention_queue = lambda: len(attention_resource.queue)

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
            attention_layers=None,  # temp, will fix after we create them
            final_completion_tracker=final_completion_tracker,
            batch_stats=batch_stats,
            defrag_v0_debug_state=defrag_v0_debug_state,
            scheduling_policy=scheduling_policy,
        )
        expert_workers.append(worker)

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
            expert_workers=expert_workers,
            profile_router=profile_router,
            attention_resource=attention_resource,
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
    print(f"Average request latency: {avg_request_latency_ms:.3f} ms")
    print(f"P90 request latency: {p90_request_latency_ms:.3f} ms")
    print(f"P99 request latency: {p99_request_latency_ms:.3f} ms")
    print(f"Average per-expert batch size: {avg_per_expert_batch_size:.3f}")

    return {
        "completion_times": completion_times,
        "avg_latency": avg_latency,
        "makespan": makespan,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "stopped_early": stopped_early,
        "steps_executed": steps_executed,
        "saturation_step": saturation_step,
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
