from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import random
import sys
import time
import atexit
from typing import Any, Dict, List
from collections.abc import Callable


# Ensure repo root is on sys.path so `disagmoe` imports work when this file is
# executed as a script (e.g. `python simulation/experiment.py sync`).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import sim_sync
import sim_tbo
try:
    import sim_async  # type: ignore
except ModuleNotFoundError:
    sim_async = None  # type: ignore


def _parse_int_list_env(name: str, default: List[int]) -> List[int]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return list(default)
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or list(default)


def _resolve_expert_cost_profile_path() -> str:
    profile = (os.environ.get("EXPERT_COST_PROFILE") or "").strip()
    if not profile:
        profile = "GPT-OSS-120B_A100.csv"
    if os.path.isabs(profile):
        return profile
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_costs_profiles", profile)


def _resolve_gating_profile_path() -> str:
    profile = (os.environ.get("GATING_PROFILE") or "").strip()
    if not profile:
        profile = "gating_gptoss120b_200.parquet"
    if os.path.isabs(profile):
        return profile
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gating_profiles", profile)


# ============================================================
# Configuration macros (all tunables gathered in one place)
# ============================================================

# ATTN_SERVICE_T_VALUES = [1, 2, 4]
ATTN_SERVICE_T_VALUES = [0.1, 0.2, 0.4]
# EP_GROUP_SIZE_VALUES = [8, 16, 32]
EP_GROUP_SIZE_VALUES = [32]
GLOBAL_BATCH_MULTIPLIERS = [128, 256]

# Simple topology model:
# - GPUs [0..N_GPU_PER_HOST-1] are on host 0, [N_GPU_PER_HOST..2*N_GPU_PER_HOST-1] on host 1, etc.
# - Network delay depends on whether src/dst GPUs are on the same host.
N_GPU_PER_HOST = 4
NET_DELAY_VALUES_INTRA_HOST = [0.04]
NET_DELAY_VALUES_INTER_HOST = [0.08]

# Toggle bandwidth-aware (congestion-modeled) network simulation.
# When True, transfer times are computed from data size and link bandwidth with
# receiver-side fair-sharing.  When False, fixed per-hop delays are used.
# Both values are swept in the experiment grid so results can be compared.
ENBALE_BANDWIDTH_UNAWARE_COMPARISON = False
ENABLE_BANDWIDTH_AWARE_NETWORK_VALUES = (
    [True, False] if ENBALE_BANDWIDTH_UNAWARE_COMPARISON else [True]
)

# ------------------------------
# Model + cost-profile macros
# ------------------------------
MODEL_NUM_LAYERS = int(os.environ.get("MODEL_NUM_LAYERS", "36") or "36")
MODEL_ROUTING_TOPK = int(os.environ.get("MODEL_ROUTING_TOPK", "4") or "4")
MODEL_NUM_EXPERTS = int(os.environ.get("MODEL_NUM_EXPERTS", "128") or "128")

HIDDEN_DIM_VALUES = _parse_int_list_env("MODEL_HIDDEN_DIMS", default=[2880])  # bytes/token for comm model
BYTES_PER_ELEMENT = 2                      # FP16 = 2 bytes
INTRA_NODE_BANDWIDTH_GBPS = 800.0          # NVLink bandwidth (Gb/s)
INTER_NODE_BANDWIDTH_VALUES = [50, 100, 200]   # InfiniBand/RoCE bandwidth (Gb/s) to sweep
TICKS_PER_MILLISECOND = 10                 # Conversion factor for simulation time

MAX_WORKERS = 56

# Enable early termination once max concurrent active requests is hit.
ENABLE_EARLY_TERMINATION_AFTER_MAX_BS = True

# Once max concurrent active requests is first reached, let this many additional
# tokens complete before stopping and reporting metrics.
TOKENS__AFTER_REACHING_MAX_BS = 8000

# Optional: record a short post-saturation schedule timeline for the TBO simulator.
# Set env var `TBO_TIMELINE_CAPTURE_OPS` (e.g. "100") to enable.
try:
    TBO_TIMELINE_CAPTURE_OPS = int(os.environ.get("TBO_TIMELINE_CAPTURE_OPS", "0") or "0")
except ValueError:
    TBO_TIMELINE_CAPTURE_OPS = 0

# Async-only: after the system first reaches max concurrent active requests, keep
# updating a single defrag-v0 scheduler debug snapshot (queue lengths + score matrix),
# and write the *last* snapshot observed before simulation termination.
ENABLE_DEFRAG_V0_DEBUG_LOG = False
DEFRAG_V0_DEBUG_LOG_BASENAME = "experiment_defrag_v0_debug_async.txt"

# Per-token stats: sampled per-token latency ("ITL") logging.
ENABLE_PER_TOKEN_STATS = True
TOKEN_SAMPLING_RATE = 0.5


_PER_TOKEN_STATS_F = None
_PER_TOKEN_STATS_RNG: random.Random | None = None
_PER_TOKEN_STATS_SAMPLING_RATE: float = 0.0


def _per_token_stats_dir() -> str:
    # Always place logs under ./simulation/per-token-stats (repo-relative),
    # regardless of the current working directory.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "per-token-stats")


def _close_per_token_stats_file() -> None:
    global _PER_TOKEN_STATS_F
    f = _PER_TOKEN_STATS_F
    _PER_TOKEN_STATS_F = None
    if f is not None:
        try:
            f.close()
        except Exception:
            pass


def _init_worker_per_token_stats(mode: str, enabled: bool, sampling_rate: float) -> None:
    global _PER_TOKEN_STATS_F, _PER_TOKEN_STATS_RNG, _PER_TOKEN_STATS_SAMPLING_RATE

    if not enabled:
        _PER_TOKEN_STATS_F = None
        _PER_TOKEN_STATS_RNG = None
        _PER_TOKEN_STATS_SAMPLING_RATE = 0.0
        return

    sampling_rate = float(sampling_rate)
    if not (0.0 <= sampling_rate <= 1.0):
        raise ValueError("TOKEN_SAMPLING_RATE must be in [0.0, 1.0]")

    os.makedirs(_per_token_stats_dir(), exist_ok=True)
    worker_name = mp.current_process().name
    out_path = os.path.join(
        _per_token_stats_dir(),
        f"per_token_stats_{mode}_{worker_name}.csv",
    )

    # Overwrite per-worker log file if it exists.
    _PER_TOKEN_STATS_F = open(out_path, "w", encoding="utf-8", newline="", buffering=1)
    _PER_TOKEN_STATS_F.write(
        "mode,worker,pid,ep_group_size,global_request_max_batch_size,attn_service_t,"
        "n_gpu_per_host,net_delay_intra_host,net_delay_inter_host,"
        "hidden_dim,inter_node_bw_gbps,bw_aware,"
        "request_id,token_index,tid,birth_time,completion_time,latency_ticks,latency_ms\n"
    )
    _PER_TOKEN_STATS_F.flush()

    seed = (os.getpid() << 16) ^ int(time.time() * 1_000_000)
    _PER_TOKEN_STATS_RNG = random.Random(seed)
    _PER_TOKEN_STATS_SAMPLING_RATE = sampling_rate

    atexit.register(_close_per_token_stats_file)


def _per_token_stats_begin_sample(token: Any) -> None:
    rng = _PER_TOKEN_STATS_RNG
    if rng is None:
        return
    rate = _PER_TOKEN_STATS_SAMPLING_RATE
    if rate <= 0.0:
        return
    sampled = True if rate >= 1.0 else (rng.random() < rate)
    try:
        setattr(token, "sampled_for_stats", bool(sampled))
    except Exception:
        pass


def _per_token_stats_log_if_sampled(
    *,
    mode: str,
    cfg: Dict[str, Any],
    token: Any,
    completion_time: float,
    ticks_per_ms: float,
) -> None:
    f = _PER_TOKEN_STATS_F
    if f is None:
        return

    if not bool(getattr(token, "sampled_for_stats", False)):
        return

    latency_ticks = completion_time - float(getattr(token, "birth_time", 0.0))
    latency_ms = latency_ticks / float(ticks_per_ms) if ticks_per_ms else 0.0

    worker_name = mp.current_process().name
    pid = os.getpid()

    f.write(
        f"{mode},"
        f"{worker_name},"
        f"{pid},"
        f"{cfg.get('ep_group_size')},"
        f"{cfg.get('global_request_max_batch_size')},"
        f"{cfg.get('attn_service_t')},"
        f"{cfg.get('n_gpu_per_host')},"
        f"{cfg.get('net_delay_intra_host')},"
        f"{cfg.get('net_delay_inter_host')},"
        f"{cfg.get('hidden_dim', 0)},"
        f"{cfg.get('inter_node_bw_gbps', 0.0)},"
        f"{cfg.get('bw_aware', True)},"
        f"{getattr(token, 'request_id', '')},"
        f"{getattr(token, 'token_index', '')},"
        f"{getattr(token, 'tid', '')},"
        f"{getattr(token, 'birth_time', '')},"
        f"{completion_time},"
        f"{latency_ticks},"
        f"{latency_ms}\n"
    )


def _format_int_matrix(matrix: List[List[int]]) -> str:
    if not matrix:
        return "<empty>\n"
    cols = max((len(r) for r in matrix), default=0)
    col_width = max(2, max((len(str(v)) for r in matrix for v in r), default=1))
    lines = []
    for i, row in enumerate(matrix):
        padded = row + [0] * max(0, cols - len(row))
        cells = " ".join(f"{v:{col_width}d}" for v in padded)
        lines.append(f"L{i:02d}: {cells}")
    return "\n".join(lines) + "\n"


def _format_optional_float_matrix(matrix: List[List[float | None]]) -> str:
    if not matrix:
        return "<empty>\n"
    cols = max((len(r) for r in matrix), default=0)
    lines = []
    for i, row in enumerate(matrix):
        padded = row + [None] * max(0, cols - len(row))
        cells = " ".join(
            (f"{v:10.3f}" if v is not None else f"{'.':>10}") for v in padded
        )
        lines.append(f"L{i:02d}: {cells}")
    return "\n".join(lines) + "\n"


def _build_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for ep_group_size, attn_t, net_delay_intra, net_delay_inter, mult, hidden_dim, inter_bw, bw_aware in itertools.product(
        EP_GROUP_SIZE_VALUES,
        ATTN_SERVICE_T_VALUES,
        NET_DELAY_VALUES_INTRA_HOST,
        NET_DELAY_VALUES_INTER_HOST,
        GLOBAL_BATCH_MULTIPLIERS,
        HIDDEN_DIM_VALUES,
        INTER_NODE_BANDWIDTH_VALUES,
        ENABLE_BANDWIDTH_AWARE_NETWORK_VALUES,
    ):
        global_batch = ep_group_size * mult
        configs.append(
            {
                "ep_group_size": ep_group_size,
                "attn_service_t": attn_t,
                "n_gpu_per_host": N_GPU_PER_HOST,
                "net_delay_intra_host": net_delay_intra,
                "net_delay_inter_host": net_delay_inter,
                "global_request_max_batch_size": global_batch,
                "hidden_dim": hidden_dim,
                "inter_node_bw_gbps": inter_bw,
                "bw_aware": bool(bw_aware),
            }
        )
    return configs


def _make_topology_net_delay_fn(
    *,
    n_gpu_per_host: int,
    net_delay_intra_host: float,
    net_delay_inter_host: float,
):
    n_gpu_per_host = max(1, int(n_gpu_per_host))
    net_delay_intra_host = float(net_delay_intra_host)
    net_delay_inter_host = float(net_delay_inter_host)

    def net_delay_fn(src_gpu: int, dst_gpu: int) -> float:
        src_host = int(src_gpu) // n_gpu_per_host
        dst_host = int(dst_gpu) // n_gpu_per_host
        return net_delay_intra_host if src_host == dst_host else net_delay_inter_host

    return net_delay_fn


def _make_bandwidth_aware_delay_fn(
    *,
    n_gpu_per_host: int,
    hidden_dim: int,
    bytes_per_elem: int = BYTES_PER_ELEMENT,
    intra_node_bw_gbps: float = INTRA_NODE_BANDWIDTH_GBPS,
    inter_node_bw_gbps: float,
    ticks_per_ms: float = TICKS_PER_MILLISECOND,
):
    """
    Create a bandwidth-aware delay function that computes transfer time
    based on the amount of data being transferred.

    Args:
        n_gpu_per_host: Number of GPUs per host for topology
        hidden_dim: Hidden dimension of the model
        bytes_per_elem: Bytes per element (2 for FP16, 4 for FP32)
        intra_node_bw_gbps: Intra-node bandwidth in GB/s
        inter_node_bw_gbps: Inter-node bandwidth in GB/s
        ticks_per_ms: Conversion factor from ms to simulation ticks

    Returns:
        A function(src_gpu, dst_gpu, num_tokens=1) -> delay_ticks
    """
    n_gpu_per_host = max(1, int(n_gpu_per_host))
    bytes_per_token = int(hidden_dim) * int(bytes_per_elem)
    intra_bw_bytes_per_ms = float(intra_node_bw_gbps) * 1e9 / 8.0 / 1000.0  # Gb/s -> bytes/ms
    inter_bw_bytes_per_ms = float(inter_node_bw_gbps) * 1e9 / 8.0 / 1000.0  # Gb/s -> bytes/ms

    def net_delay_fn(src_gpu: int, dst_gpu: int, num_tokens: int = 1) -> float:
        src_host = int(src_gpu) // n_gpu_per_host
        dst_host = int(dst_gpu) // n_gpu_per_host

        data_bytes = float(num_tokens) * float(bytes_per_token)
        bandwidth = intra_bw_bytes_per_ms if src_host == dst_host else inter_bw_bytes_per_ms

        transfer_time_ms = data_bytes / bandwidth
        transfer_time_ticks = transfer_time_ms * float(ticks_per_ms)
        return transfer_time_ticks

    return net_delay_fn


def _run_async_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if sim_async is None:
        raise RuntimeError(
            "Async simulator dependencies not available. "
            "Install `simpy` (and other requirements) or run `sync`/`tbo` modes instead."
        )
    # Configure per-run timing parameters via module globals,
    # then call the async simulator with explicit ep_group_size and batch size.
    sim_async.ATTN_SERVICE_T = cfg["attn_service_t"]
    # Keep module globals set for backward-compat / defaults; the actual delay
    # applied in the simulator is driven by `net_delay_fn` below.
    sim_async.NET_T_ATTN_TO_EXPERT = cfg["net_delay_inter_host"]
    sim_async.NET_T_EXPERT_TO_ATTN = cfg["net_delay_inter_host"]

    net_delay_fn = None
    if cfg.get("bw_aware", True):
        net_delay_fn = _make_bandwidth_aware_delay_fn(
            n_gpu_per_host=cfg["n_gpu_per_host"],
            hidden_dim=cfg["hidden_dim"],
            bytes_per_elem=BYTES_PER_ELEMENT,
            intra_node_bw_gbps=INTRA_NODE_BANDWIDTH_GBPS,
            inter_node_bw_gbps=cfg["inter_node_bw_gbps"],
            ticks_per_ms=sim_async.TICKS_PER_MILLISECOND,
        )

    tokens_after_reaching_max_bs = (
        TOKENS__AFTER_REACHING_MAX_BS if ENABLE_EARLY_TERMINATION_AFTER_MAX_BS else None
    )
    per_token_begin_cb: Callable[[Any], None] | None = None
    per_token_stats_cb: Callable[[Any, float], None] | None = None
    if ENABLE_PER_TOKEN_STATS:
        ticks_per_ms = sim_async.TICKS_PER_MILLISECOND
        per_token_begin_cb = _per_token_stats_begin_sample

        def _per_token_stats_cb(token: Any, t_complete: float) -> None:
            _per_token_stats_log_if_sampled(
                mode="async",
                cfg=cfg,
                token=token,
                completion_time=t_complete,
                ticks_per_ms=ticks_per_ms,
            )

        per_token_stats_cb = _per_token_stats_cb

    result = sim_async.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
        attn_dp_group_size=cfg["ep_group_size"],
        total_expert_count=MODEL_NUM_EXPERTS,
        num_layers=MODEL_NUM_LAYERS,
        routing_top_k=MODEL_ROUTING_TOPK,
        expert_profile_path=_resolve_expert_cost_profile_path(),
        profile_routing_path=_resolve_gating_profile_path(),
        tokens_after_reaching_max_bs=tokens_after_reaching_max_bs,
        enable_defrag_v0_debug_log=ENABLE_DEFRAG_V0_DEBUG_LOG,
        per_token_begin_cb=per_token_begin_cb,
        per_token_stats_cb=per_token_stats_cb,
        net_delay_fn=net_delay_fn,
        n_gpu_per_host=cfg["n_gpu_per_host"],
    )

    ticks_per_ms = sim_async.TICKS_PER_MILLISECOND
    avg_latency_ms = (
        result["avg_latency"] / ticks_per_ms if result["avg_latency"] else 0.0
    )
    makespan_ms = (
        result["makespan"] / ticks_per_ms if result["makespan"] else 0.0
    )

    avg_per_expert_batch_size = result.get("avg_per_expert_batch_size", 0.0)
    tail_throughput_req_per_sec = result.get(
        "tail_throughput_req_per_sec", float("nan")
    )

    debug_text = None
    if ENABLE_DEFRAG_V0_DEBUG_LOG and result.get("defrag_v0_debug_triggered"):
        entry = result.get("defrag_v0_debug_entry") or {}
        queue_len_matrix = entry.get("queue_len_matrix") or []
        score_matrix = entry.get("score_matrix") or []
        layer_totals = entry.get("layer_totals")
        lookahead_scores_by_layer = entry.get("lookahead_scores_by_layer")

        debug_lines = [
            "=== defrag_v0_debug (async) ===",
            f"config: {cfg}",
            f"trigger_time={result.get('defrag_v0_debug_trigger_time')}, "
            f"log_time={entry.get('log_time')}, worker_idx={entry.get('worker_idx')}",
            f"active_requests_at_log_time={entry.get('active_requests_at_log_time')}, "
            f"pending_queue_len_at_log_time={entry.get('pending_queue_len_at_log_time')}",
            f"global_expert_queued_tokens={entry.get('global_expert_queued_tokens')}, "
            f"global_expert_inflight_tokens={entry.get('global_expert_inflight_tokens')}, "
            f"global_expert_total_tokens={entry.get('global_expert_total_tokens')}",
            f"attention_users={entry.get('attention_users')}, attention_queue={entry.get('attention_queue')}",
            f"pending_tokens_total={entry.get('pending_tokens_total')}",
            f"pending_tokens_per_layer={entry.get('pending_tokens_per_layer')}",
            f"tokens_created={entry.get('tokens_created')}, tokens_completed={entry.get('tokens_completed')}, "
            f"tokens_inflight_unique={entry.get('tokens_inflight_unique')}",
            f"per_worker_expert_queued_tokens={entry.get('per_worker_expert_queued_tokens')}",
            f"per_worker_expert_inflight_tokens={entry.get('per_worker_expert_inflight_tokens')}",
            f"params: weight_decay={entry.get('weight_decay')}, lookahead_steps={entry.get('lookahead_steps')}",
        ]
        if layer_totals is not None:
            debug_lines.append(f"layer_totals: {layer_totals}")
        if lookahead_scores_by_layer is not None:
            debug_lines.append(f"lookahead_scores_by_layer: {lookahead_scores_by_layer}")
        debug_lines.append("")
        debug_lines.append("queue_len_matrix (layers x groups):")
        debug_lines.append(_format_int_matrix(queue_len_matrix).rstrip("\n"))
        debug_lines.append("")
        debug_lines.append("score_matrix (layers x groups, '.' = empty queue):")
        debug_lines.append(_format_optional_float_matrix(score_matrix).rstrip("\n"))
        debug_lines.append("")
        debug_lines.append(
            "best: "
            f"layer={entry.get('best_layer_idx')}, "
            f"group={entry.get('best_group_idx')}, "
            f"score={entry.get('best_score')}"
        )
        debug_lines.append("")
        debug_text = "\n".join(debug_lines)

    summary: Dict[str, Any] = {
        "mode": "async",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "n_gpu_per_host": cfg["n_gpu_per_host"],
        "net_delay_intra_host": cfg["net_delay_intra_host"],
        "net_delay_inter_host": cfg["net_delay_inter_host"],
        "hidden_dim": cfg.get("hidden_dim", 0),
        "inter_node_bw_gbps": cfg.get("inter_node_bw_gbps", 0),
        "bw_aware": cfg.get("bw_aware", True),
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": result.get("avg_throughput_req_per_sec", 0.0),
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": result.get("avg_request_latency_ms", 0.0),
        "p90_request_latency_ms": result.get("p90_request_latency_ms", 0.0),
        "p99_request_latency_ms": result.get("p99_request_latency_ms", 0.0),
        # Not applicable for async simulator; keep for uniform schema.
        "avg_layer_runtime_ms": float("nan"),
        "avg_layer_wait_imbalance_ms": float("nan"),
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "avg_layer_worker_queue_stddev": float("nan"),
        "defrag_v0_debug_text": debug_text,
    }
    return summary


def _run_sync_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tokens_after_reaching_max_bs = (
        TOKENS__AFTER_REACHING_MAX_BS if ENABLE_EARLY_TERMINATION_AFTER_MAX_BS else None
    )
    per_token_begin_cb: Callable[[Any], None] | None = None
    per_token_stats_cb: Callable[[Any, float], None] | None = None
    if ENABLE_PER_TOKEN_STATS:
        ticks_per_ms = sim_sync.TICKS_PER_MILLISECOND
        per_token_begin_cb = _per_token_stats_begin_sample

        def _per_token_stats_cb(token: Any, t_complete: float) -> None:
            _per_token_stats_log_if_sampled(
                mode="sync",
                cfg=cfg,
                token=token,
                completion_time=t_complete,
                ticks_per_ms=ticks_per_ms,
            )

        per_token_stats_cb = _per_token_stats_cb

    net_delay_fn = None
    if cfg.get("bw_aware", True):
        net_delay_fn = _make_bandwidth_aware_delay_fn(
            n_gpu_per_host=cfg["n_gpu_per_host"],
            hidden_dim=cfg["hidden_dim"],
            bytes_per_elem=BYTES_PER_ELEMENT,
            intra_node_bw_gbps=INTRA_NODE_BANDWIDTH_GBPS,
            inter_node_bw_gbps=cfg["inter_node_bw_gbps"],
            ticks_per_ms=sim_sync.TICKS_PER_MILLISECOND,
        )

    result = sim_sync.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        attn_service_t=cfg["attn_service_t"],
        attn_dp_group_size=cfg["ep_group_size"],
        net_t_attn_to_expert=cfg["net_delay_inter_host"],
        net_t_expert_to_attn=cfg["net_delay_inter_host"],
        total_expert_count=MODEL_NUM_EXPERTS,
        num_layers=MODEL_NUM_LAYERS,
        routing_top_k=MODEL_ROUTING_TOPK,
        expert_profile_path=_resolve_expert_cost_profile_path(),
        profile_routing_path=_resolve_gating_profile_path(),
        net_delay_fn=net_delay_fn,
        n_gpu_per_host=cfg["n_gpu_per_host"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
        tokens_after_reaching_max_bs=tokens_after_reaching_max_bs,
        per_token_begin_cb=per_token_begin_cb,
        per_token_stats_cb=per_token_stats_cb,
    )

    ticks_per_ms = sim_sync.TICKS_PER_MILLISECOND
    avg_latency_ms = (
        result["avg_latency"] / ticks_per_ms if result["avg_latency"] else 0.0
    )
    makespan_ms = (
        result["makespan"] / ticks_per_ms if result["makespan"] else 0.0
    )

    avg_layer_runtime_ms = result.get("avg_layer_runtime", 0.0) / ticks_per_ms
    avg_layer_wait_imbalance_ms = (
        result.get("avg_layer_wait_imbalance", 0.0) / ticks_per_ms
    )

    avg_per_expert_batch_size = result.get("avg_per_expert_batch_size", 0.0)
    avg_layer_worker_queue_stddev = result.get(
        "avg_layer_worker_queue_stddev", 0.0
    )

    avg_throughput_req_per_sec = result.get("avg_throughput_req_per_sec", 0.0)
    tail_throughput_req_per_sec = result.get(
        "tail_throughput_req_per_sec", float("nan")
    )
    avg_request_latency_ms = result.get("avg_request_latency_ms", 0.0)
    p90_request_latency_ms = result.get("p90_request_latency_ms", 0.0)
    p99_request_latency_ms = result.get("p99_request_latency_ms", 0.0)

    summary: Dict[str, Any] = {
        "mode": "sync",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "n_gpu_per_host": cfg["n_gpu_per_host"],
        "net_delay_intra_host": cfg["net_delay_intra_host"],
        "net_delay_inter_host": cfg["net_delay_inter_host"],
        "hidden_dim": cfg.get("hidden_dim", 0),
        "inter_node_bw_gbps": cfg.get("inter_node_bw_gbps", 0),
        "bw_aware": cfg.get("bw_aware", True),
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
        "avg_layer_runtime_ms": avg_layer_runtime_ms,
        "avg_layer_wait_imbalance_ms": avg_layer_wait_imbalance_ms,
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "avg_layer_worker_queue_stddev": avg_layer_worker_queue_stddev,
    }
    return summary


def _run_tbo_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tokens_after_reaching_max_bs = (
        TOKENS__AFTER_REACHING_MAX_BS if ENABLE_EARLY_TERMINATION_AFTER_MAX_BS else None
    )
    per_token_begin_cb: Callable[[Any], None] | None = None
    per_token_stats_cb: Callable[[Any, float], None] | None = None
    if ENABLE_PER_TOKEN_STATS:
        ticks_per_ms = sim_tbo.TICKS_PER_MILLISECOND
        per_token_begin_cb = _per_token_stats_begin_sample

        def _per_token_stats_cb(token: Any, t_complete: float) -> None:
            _per_token_stats_log_if_sampled(
                mode="tbo",
                cfg=cfg,
                token=token,
                completion_time=t_complete,
                ticks_per_ms=ticks_per_ms,
            )

        per_token_stats_cb = _per_token_stats_cb

    net_delay_fn = None
    if cfg.get("bw_aware", True):
        net_delay_fn = _make_bandwidth_aware_delay_fn(
            n_gpu_per_host=cfg["n_gpu_per_host"],
            hidden_dim=cfg["hidden_dim"],
            bytes_per_elem=BYTES_PER_ELEMENT,
            intra_node_bw_gbps=INTRA_NODE_BANDWIDTH_GBPS,
            inter_node_bw_gbps=cfg["inter_node_bw_gbps"],
            ticks_per_ms=sim_tbo.TICKS_PER_MILLISECOND,
        )

    result = sim_tbo.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        attn_service_t=cfg["attn_service_t"],
        attn_dp_group_size=cfg["ep_group_size"],
        net_t_attn_to_expert=cfg["net_delay_inter_host"],
        net_t_expert_to_attn=cfg["net_delay_inter_host"],
        total_expert_count=MODEL_NUM_EXPERTS,
        num_layers=MODEL_NUM_LAYERS,
        routing_top_k=MODEL_ROUTING_TOPK,
        expert_profile_path=_resolve_expert_cost_profile_path(),
        profile_routing_path=_resolve_gating_profile_path(),
        net_delay_fn=net_delay_fn,
        n_gpu_per_host=cfg["n_gpu_per_host"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
        tokens_after_reaching_max_bs=tokens_after_reaching_max_bs,
        per_token_begin_cb=per_token_begin_cb,
        per_token_stats_cb=per_token_stats_cb,
        timeline_capture_ops=TBO_TIMELINE_CAPTURE_OPS,
        timeline_out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tbo-timeline"),
        timeline_basename=(
            f"tbo_timeline_ep{cfg['ep_group_size']}"
            f"_gbs{cfg['global_request_max_batch_size']}"
            f"_attn{cfg['attn_service_t']}"
            f"_pid{os.getpid()}"
        ),
    )

    ticks_per_ms = sim_tbo.TICKS_PER_MILLISECOND
    avg_latency_ms = (
        result["avg_latency"] / ticks_per_ms if result["avg_latency"] else 0.0
    )
    makespan_ms = (
        result["makespan"] / ticks_per_ms if result["makespan"] else 0.0
    )

    avg_layer_runtime_ms = result.get("avg_layer_runtime", 0.0) / ticks_per_ms
    avg_layer_wait_imbalance_ms = (
        result.get("avg_layer_wait_imbalance", 0.0) / ticks_per_ms
    )

    avg_per_expert_batch_size = result.get("avg_per_expert_batch_size", 0.0)
    avg_layer_worker_queue_stddev = result.get(
        "avg_layer_worker_queue_stddev", 0.0
    )

    avg_throughput_req_per_sec = result.get("avg_throughput_req_per_sec", 0.0)
    tail_throughput_req_per_sec = result.get(
        "tail_throughput_req_per_sec", float("nan")
    )
    avg_request_latency_ms = result.get("avg_request_latency_ms", 0.0)
    p90_request_latency_ms = result.get("p90_request_latency_ms", 0.0)
    p99_request_latency_ms = result.get("p99_request_latency_ms", 0.0)

    summary: Dict[str, Any] = {
        "mode": "tbo",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "n_gpu_per_host": cfg["n_gpu_per_host"],
        "net_delay_intra_host": cfg["net_delay_intra_host"],
        "net_delay_inter_host": cfg["net_delay_inter_host"],
        "hidden_dim": cfg.get("hidden_dim", 0),
        "inter_node_bw_gbps": cfg.get("inter_node_bw_gbps", 0),
        "bw_aware": cfg.get("bw_aware", True),
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
        "avg_layer_runtime_ms": avg_layer_runtime_ms,
        "avg_layer_wait_imbalance_ms": avg_layer_wait_imbalance_ms,
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "avg_layer_worker_queue_stddev": avg_layer_worker_queue_stddev,
    }
    return summary


def _write_results(mode: str, results: List[Dict[str, Any]]) -> None:
    results_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(results_dir, f"experiment_results_{mode}.txt")

    header = (
        "mode,ep_group_size,global_request_max_batch_size,"
        "attn_service_t,n_gpu_per_host,net_delay_intra_host,net_delay_inter_host,"
        "hidden_dim,inter_node_bw_gbps,bw_aware,"
        "avg_token_latency_ms,makespan_ms,avg_throughput_req_per_sec,tail_throughput_req_per_sec,"
        "avg_request_latency_ms,p90_request_latency_ms,p99_request_latency_ms,"
        "avg_layer_runtime_ms,avg_layer_wait_imbalance_ms,"
        "avg_per_expert_batch_size,avg_layer_worker_queue_stddev\n"
    )

    # Sort results for reproducible ordering.
    results_sorted = sorted(
        results,
        key=lambda r: (
            r["ep_group_size"],
            r["global_request_max_batch_size"],
            r["attn_service_t"],
            r["n_gpu_per_host"],
            r["net_delay_intra_host"],
            r["net_delay_inter_host"],
            r.get("hidden_dim", 0),
            r.get("inter_node_bw_gbps", 0),
            0 if r.get("bw_aware", True) else 1,
        ),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in results_sorted:
            line = (
                f"{r['mode']},"
                f"{r['ep_group_size']},"
                f"{r['global_request_max_batch_size']},"
                f"{r['attn_service_t']},"
                f"{r['n_gpu_per_host']},"
                f"{r['net_delay_intra_host']},"
                f"{r['net_delay_inter_host']},"
                f"{r.get('hidden_dim', '')},"
                f"{r.get('inter_node_bw_gbps', '')},"
                f"{r.get('bw_aware', True)},"
                f"{r['avg_token_latency_ms']:.6f},"
                f"{r['makespan_ms']:.6f},"
                f"{r['avg_throughput_req_per_sec']:.6f},"
                f"{r.get('tail_throughput_req_per_sec', float('nan')):.6f},"
                f"{r['avg_request_latency_ms']:.6f},"
                f"{r['p90_request_latency_ms']:.6f},"
                f"{r['p99_request_latency_ms']:.6f},"
                f"{r['avg_layer_runtime_ms']:.6f},"
                f"{r['avg_layer_wait_imbalance_ms']:.6f},"
                f"{r['avg_per_expert_batch_size']:.6f},"
                f"{r['avg_layer_worker_queue_stddev']:.6f}\n"
            )
            f.write(line)

    print(f"Wrote results for mode='{mode}' to {out_path}")


def main(argv: List[str]) -> None:
    if len(argv) != 2 or argv[1] not in ("sync", "async", "tbo"):
        print("Usage: python experiment.py [sync|async|tbo]")
        raise SystemExit(1)

    mode = argv[1]

    # Print resolved macros once per invocation.
    profile_path = _resolve_expert_cost_profile_path()
    gating_profile_path = _resolve_gating_profile_path()
    print(
        "Experiment macros: "
        f"MODEL_NUM_EXPERTS={MODEL_NUM_EXPERTS} "
        f"MODEL_NUM_LAYERS={MODEL_NUM_LAYERS} "
        f"MODEL_ROUTING_TOPK={MODEL_ROUTING_TOPK} "
        f"ENBALE_BANDWIDTH_UNAWARE_COMPARISON={ENBALE_BANDWIDTH_UNAWARE_COMPARISON} "
        f"MODEL_HIDDEN_DIMS={HIDDEN_DIM_VALUES} "
        f"EXPERT_COST_PROFILE={profile_path} "
        f"GATING_PROFILE={gating_profile_path}"
    )
    configs = _build_configs()
    num_workers = min(MAX_WORKERS, len(configs))

    if mode == "async":
        worker_fn = _run_async_config
    elif mode == "sync":
        worker_fn = _run_sync_config
    else:
        worker_fn = _run_tbo_config

    results_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(results_dir, f"experiment_results_{mode}.txt")
    debug_path = os.path.join(results_dir, DEFRAG_V0_DEBUG_LOG_BASENAME)

    header = (
        "mode,ep_group_size,global_request_max_batch_size,"
        "attn_service_t,n_gpu_per_host,net_delay_intra_host,net_delay_inter_host,"
        "hidden_dim,inter_node_bw_gbps,bw_aware,"
        "avg_token_latency_ms,makespan_ms,avg_throughput_req_per_sec,tail_throughput_req_per_sec,"
        "avg_request_latency_ms,p90_request_latency_ms,p99_request_latency_ms,"
        "avg_layer_runtime_ms,avg_layer_wait_imbalance_ms,"
        "avg_per_expert_batch_size,avg_layer_worker_queue_stddev\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        debug_f = None
        try:
            if mode == "async" and ENABLE_DEFRAG_V0_DEBUG_LOG:
                debug_f = open(debug_path, "w", encoding="utf-8")
                debug_f.flush()

            if ENABLE_PER_TOKEN_STATS:
                os.makedirs(_per_token_stats_dir(), exist_ok=True)
                prefix = f"per_token_stats_{mode}_"
                for name in os.listdir(_per_token_stats_dir()):
                    if name.startswith(prefix):
                        try:
                            os.remove(os.path.join(_per_token_stats_dir(), name))
                        except FileNotFoundError:
                            pass

            pool_kwargs: Dict[str, Any] = {"processes": num_workers}
            if ENABLE_PER_TOKEN_STATS:
                pool_kwargs["initializer"] = _init_worker_per_token_stats
                pool_kwargs["initargs"] = (mode, True, TOKEN_SAMPLING_RATE)

            with mp.Pool(**pool_kwargs) as pool:
                for summary in pool.imap_unordered(worker_fn, configs):
                    line = (
                        f"{summary['mode']},"
                        f"{summary['ep_group_size']},"
                        f"{summary['global_request_max_batch_size']},"
                        f"{summary['attn_service_t']},"
                        f"{summary['n_gpu_per_host']},"
                        f"{summary['net_delay_intra_host']},"
                        f"{summary['net_delay_inter_host']},"
                        f"{summary.get('hidden_dim', '')},"
                        f"{summary.get('inter_node_bw_gbps', '')},"
                        f"{summary.get('bw_aware', True)},"
                        f"{summary['avg_token_latency_ms']:.6f},"
                        f"{summary['makespan_ms']:.6f},"
                        f"{summary['avg_throughput_req_per_sec']:.6f},"
                        f"{summary.get('tail_throughput_req_per_sec', float('nan')):.6f},"
                        f"{summary['avg_request_latency_ms']:.6f},"
                        f"{summary['p90_request_latency_ms']:.6f},"
                        f"{summary['p99_request_latency_ms']:.6f},"
                        f"{summary['avg_layer_runtime_ms']:.6f},"
                        f"{summary['avg_layer_wait_imbalance_ms']:.6f},"
                        f"{summary['avg_per_expert_batch_size']:.6f},"
                        f"{summary['avg_layer_worker_queue_stddev']:.6f}\n"
                    )
                    f.write(line)
                    f.flush()

                    if debug_f is not None:
                        dbg = summary.get("defrag_v0_debug_text")
                        if dbg:
                            debug_f.write(dbg + "\n")
                            debug_f.flush()
        finally:
            if debug_f is not None:
                debug_f.close()

    print(f"Wrote results for mode='{mode}' to {out_path}")
    if mode == "async" and ENABLE_DEFRAG_V0_DEBUG_LOG:
        print(f"Wrote defrag-v0 debug log (text) to {debug_path}")
    if ENABLE_PER_TOKEN_STATS:
        print(f"Wrote per-token sampled ITL logs under {_per_token_stats_dir()}")


if __name__ == "__main__":
    main(sys.argv)
