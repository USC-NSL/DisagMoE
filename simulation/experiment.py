from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import sys
from typing import Any, Dict, List

import sim_async
import sim_sync


ATTN_SERVICE_T_VALUES = [1, 2, 4]
EP_GROUP_SIZE_VALUES = [8, 16, 32]
GLOBAL_BATCH_MULTIPLIERS = [128, 256, 512]
NET_DELAY_VALUES = [0.1, 0.4]

MAX_WORKERS = 56


def _build_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for ep_group_size, attn_t, net_delay, mult in itertools.product(
        EP_GROUP_SIZE_VALUES,
        ATTN_SERVICE_T_VALUES,
        NET_DELAY_VALUES,
        GLOBAL_BATCH_MULTIPLIERS,
    ):
        global_batch = ep_group_size * mult
        configs.append(
            {
                "ep_group_size": ep_group_size,
                "attn_service_t": attn_t,
                "net_delay": net_delay,
                "global_request_max_batch_size": global_batch,
            }
        )
    return configs


def _run_async_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Configure per-run timing parameters via module globals,
    # then call the async simulator with explicit ep_group_size and batch size.
    sim_async.ATTN_SERVICE_T = cfg["attn_service_t"]
    sim_async.NET_T_ATTN_TO_EXPERT = cfg["net_delay"]
    sim_async.NET_T_EXPERT_TO_ATTN = cfg["net_delay"]
    result = sim_async.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
        attn_dp_group_size=cfg["ep_group_size"],
    )

    ticks_per_ms = sim_async.TICKS_PER_MILLISECOND
    avg_latency_ms = (
        result["avg_latency"] / ticks_per_ms if result["avg_latency"] else 0.0
    )
    makespan_ms = (
        result["makespan"] / ticks_per_ms if result["makespan"] else 0.0
    )

    avg_per_expert_batch_size = result.get("avg_per_expert_batch_size", 0.0)

    summary: Dict[str, Any] = {
        "mode": "async",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "net_delay": cfg["net_delay"],
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": result.get("avg_throughput_req_per_sec", 0.0),
        "avg_request_latency_ms": result.get("avg_request_latency_ms", 0.0),
        "p90_request_latency_ms": result.get("p90_request_latency_ms", 0.0),
        "p99_request_latency_ms": result.get("p99_request_latency_ms", 0.0),
        # Not applicable for async simulator; keep for uniform schema.
        "avg_layer_runtime_ms": float("nan"),
        "avg_layer_wait_imbalance_ms": float("nan"),
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "avg_layer_worker_queue_stddev": float("nan"),
    }
    return summary


def _run_sync_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    result = sim_sync.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        attn_service_t=cfg["attn_service_t"],
        attn_dp_group_size=cfg["ep_group_size"],
        net_t_attn_to_expert=cfg["net_delay"],
        net_t_expert_to_attn=cfg["net_delay"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
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
    avg_request_latency_ms = result.get("avg_request_latency_ms", 0.0)
    p90_request_latency_ms = result.get("p90_request_latency_ms", 0.0)
    p99_request_latency_ms = result.get("p99_request_latency_ms", 0.0)

    summary: Dict[str, Any] = {
        "mode": "sync",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "net_delay": cfg["net_delay"],
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
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
        "attn_service_t,net_delay,"
        "avg_token_latency_ms,makespan_ms,avg_throughput_req_per_sec,"
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
            r["net_delay"],
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
                f"{r['net_delay']},"
                f"{r['avg_token_latency_ms']:.6f},"
                f"{r['makespan_ms']:.6f},"
                f"{r['avg_throughput_req_per_sec']:.6f},"
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
    if len(argv) != 2 or argv[1] not in ("sync", "async"):
        print("Usage: python experiment.py [sync|async]")
        raise SystemExit(1)

    mode = argv[1]
    configs = _build_configs()
    num_workers = min(MAX_WORKERS, len(configs))

    worker_fn = _run_async_config if mode == "async" else _run_sync_config

    results_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(results_dir, f"experiment_results_{mode}.txt")

    header = (
        "mode,ep_group_size,global_request_max_batch_size,"
        "attn_service_t,net_delay,"
        "avg_token_latency_ms,makespan_ms,avg_throughput_req_per_sec,"
        "avg_request_latency_ms,p90_request_latency_ms,p99_request_latency_ms,"
        "avg_layer_runtime_ms,avg_layer_wait_imbalance_ms,"
        "avg_per_expert_batch_size,avg_layer_worker_queue_stddev\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        with mp.Pool(processes=num_workers) as pool:
            for summary in pool.imap_unordered(worker_fn, configs):
                line = (
                    f"{summary['mode']},"
                    f"{summary['ep_group_size']},"
                    f"{summary['global_request_max_batch_size']},"
                    f"{summary['attn_service_t']},"
                    f"{summary['net_delay']},"
                    f"{summary['avg_token_latency_ms']:.6f},"
                    f"{summary['makespan_ms']:.6f},"
                    f"{summary['avg_throughput_req_per_sec']:.6f},"
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

    print(f"Wrote results for mode='{mode}' to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
