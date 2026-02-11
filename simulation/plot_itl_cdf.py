from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ConfigKey:
    ep_group_size: int
    global_request_max_batch_size: int
    attn_service_t: float
    n_gpu_per_host: int
    net_delay_intra_host: float | None
    net_delay_inter_host: float | None
    hidden_dim: int
    inter_node_bw_gbps: float
    bw_aware: bool


@dataclass
class ExperimentMetrics:
    """Metrics from experiment_results_*.txt files."""
    avg_throughput_req_per_sec: float
    tail_throughput_req_per_sec: float
    avg_token_latency_ms: float
    avg_per_expert_batch_size: float


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _is_finite(x: float) -> bool:
    return x == x and math.isfinite(x)


def _fmt_float_for_filename(x: float) -> str:
    if not _is_finite(x):
        return "nan"
    s = f"{x:g}"
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def _iter_csv_rows(in_dir: str) -> Iterable[Tuple[str, Dict[str, str]]]:
    pattern = os.path.join(in_dir, "*.csv")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield path, row


def _key_from_row(row: Dict[str, str]) -> tuple[str, ConfigKey]:
    mode = (row.get("mode") or "").strip() or "unknown"
    attn_service_t = _safe_float(row.get("attn_service_t"), float("nan"))
    if not _is_finite(attn_service_t):
        attn_service_t = 0.0

    n_gpu_per_host = _safe_int(row.get("n_gpu_per_host"), 0)

    intra = _safe_float(row.get("net_delay_intra_host"), float("nan"))
    inter = _safe_float(row.get("net_delay_inter_host"), float("nan"))

    net_delay = _safe_float(row.get("net_delay"), float("nan"))
    if not (_is_finite(intra) and _is_finite(inter)) and _is_finite(net_delay):
        intra = net_delay
        inter = net_delay

    intra_opt = float(intra) if _is_finite(intra) else None
    inter_opt = float(inter) if _is_finite(inter) else None

    hidden_dim = _safe_int(row.get("hidden_dim"), 0)
    inter_node_bw = _safe_float(row.get("inter_node_bw_gbps"), 0.0)
    if not _is_finite(inter_node_bw):
        inter_node_bw = 0.0

    bw_aware_raw = (row.get("bw_aware") or "True").strip()
    bw_aware = bw_aware_raw not in ("False", "false", "0", "")

    return mode, ConfigKey(
        ep_group_size=_safe_int(row.get("ep_group_size"), 0),
        global_request_max_batch_size=_safe_int(
            row.get("global_request_max_batch_size"), 0
        ),
        attn_service_t=float(attn_service_t),
        n_gpu_per_host=int(n_gpu_per_host),
        net_delay_intra_host=intra_opt,
        net_delay_inter_host=inter_opt,
        hidden_dim=int(hidden_dim),
        inter_node_bw_gbps=float(inter_node_bw),
        bw_aware=bw_aware,
    )


def _extract_latency(row: Dict[str, str], unit: str) -> float | None:
    if unit == "ms":
        val = _safe_float(row.get("latency_ms"), float("nan"))
    elif unit == "ticks":
        val = _safe_float(row.get("latency_ticks"), float("nan"))
    else:
        raise ValueError("unit must be 'ms' or 'ticks'")
    return val if _is_finite(val) else None


def _load_experiment_results(
    sim_dir: str,
) -> Dict[str, Dict[ConfigKey, ExperimentMetrics]]:
    """
    Load experiment_results_*.txt files and return metrics indexed by mode and ConfigKey.
    """
    results: Dict[str, Dict[ConfigKey, ExperimentMetrics]] = {}
    for mode in ("async", "sync", "tbo"):
        path = os.path.join(sim_dir, f"experiment_results_{mode}.txt")
        if not os.path.exists(path):
            continue
        mode_results: Dict[ConfigKey, ExperimentMetrics] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _, key = _key_from_row(row)
                metrics = ExperimentMetrics(
                    avg_throughput_req_per_sec=_safe_float(
                        row.get("avg_throughput_req_per_sec"), float("nan")
                    ),
                    tail_throughput_req_per_sec=_safe_float(
                        row.get("tail_throughput_req_per_sec"), float("nan")
                    ),
                    avg_token_latency_ms=_safe_float(
                        row.get("avg_token_latency_ms"), float("nan")
                    ),
                    avg_per_expert_batch_size=_safe_float(
                        row.get("avg_per_expert_batch_size"), float("nan")
                    ),
                )
                mode_results[key] = metrics
        if mode_results:
            results[mode] = mode_results
    return results


def _print_stats_summary(
    latencies_by_cfg: Dict[ConfigKey, Dict[str, List[float]]],
    experiment_results: Dict[str, Dict[ConfigKey, ExperimentMetrics]],
    min_samples: int,
    unit: str,
) -> None:
    """Print a formatted stats summary table to stdout."""
    print("\n" + "=" * 120)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 120)

    # Collect all configs
    all_keys = sorted(
        latencies_by_cfg.keys(),
        key=lambda k: (
            k.ep_group_size,
            k.global_request_max_batch_size,
            k.attn_service_t,
            k.hidden_dim,
            k.inter_node_bw_gbps,
            0 if k.bw_aware else 1,
        ),
    )

    if not all_keys:
        print("No data available.")
        return

    # Print header
    print(f"\n{'Config':<50} | {'Mode':<6} | {'Throughput':>12} | {'Tail Tput':>12} | "
          f"{'Avg ITL':>10} | {'P95 ITL':>10} | {'Batch Size':>10} | {'Samples':>8}")
    print("-" * 130)

    for key in all_keys:
        by_mode = latencies_by_cfg.get(key, {})

        # Build config label
        if key.net_delay_intra_host is None and key.net_delay_inter_host is None:
            net_label = "net=?"
        elif key.net_delay_intra_host == key.net_delay_inter_host:
            net_label = f"net={key.net_delay_intra_host:g}" if key.net_delay_intra_host is not None else "net=?"
        else:
            intra = key.net_delay_intra_host if key.net_delay_intra_host is not None else 0
            inter = key.net_delay_inter_host if key.net_delay_inter_host is not None else 0
            net_label = f"net={intra:g}/{inter:g}"

        config_label = (
            f"ep{key.ep_group_size} gb{key.global_request_max_batch_size} "
            f"attn{key.attn_service_t:g} hid{key.hidden_dim} bw{int(key.inter_node_bw_gbps)}"
            f"{' bwa' if key.bw_aware else ' nobw'}"
        )

        first_mode = True
        for mode in ("async", "sync", "tbo"):
            values = by_mode.get(mode, [])
            n_samples = len(values)

            # Get experiment metrics
            exp_metrics = experiment_results.get(mode, {}).get(key)

            if n_samples < min_samples and exp_metrics is None:
                continue

            throughput = exp_metrics.avg_throughput_req_per_sec if exp_metrics else float("nan")
            tail_tput = exp_metrics.tail_throughput_req_per_sec if exp_metrics else float("nan")
            batch_size = exp_metrics.avg_per_expert_batch_size if exp_metrics else float("nan")

            avg_itl = _mean(values) if n_samples >= min_samples else float("nan")
            p95_itl = _p95(values) if n_samples >= min_samples else float("nan")

            cfg_col = config_label if first_mode else ""
            first_mode = False

            tput_str = f"{throughput:.2f}" if _is_finite(throughput) else "N/A"
            tail_str = f"{tail_tput:.2f}" if _is_finite(tail_tput) else "N/A"
            avg_str = f"{avg_itl:.2f}" if _is_finite(avg_itl) else "N/A"
            p95_str = f"{p95_itl:.2f}" if _is_finite(p95_itl) else "N/A"
            batch_str = f"{batch_size:.1f}" if _is_finite(batch_size) else "N/A"

            print(f"{cfg_col:<45} | {mode:<6} | {tput_str:>12} | {tail_str:>12} | "
                  f"{avg_str:>10} | {p95_str:>10} | {batch_str:>10} | {n_samples:>8}")

        print("-" * 120)

    # Print overall averages
    print("\nOVERALL AVERAGES (across all configurations)")
    print("=" * 100)
    print(f"{'Metric':<35} | {'async':>15} | {'sync':>15} | {'tbo':>15}")
    print("-" * 100)

    for mode in ("async", "sync", "tbo"):
        mode_metrics = experiment_results.get(mode, {})
        if not mode_metrics:
            continue

    # Collect averages per mode
    mode_avgs: Dict[str, Dict[str, float]] = {}
    for mode in ("async", "sync", "tbo"):
        mode_metrics = experiment_results.get(mode, {})
        if not mode_metrics:
            continue
        tputs = [m.avg_throughput_req_per_sec for m in mode_metrics.values() if _is_finite(m.avg_throughput_req_per_sec)]
        tail_tputs = [m.tail_throughput_req_per_sec for m in mode_metrics.values() if _is_finite(m.tail_throughput_req_per_sec)]
        latencies = [m.avg_token_latency_ms for m in mode_metrics.values() if _is_finite(m.avg_token_latency_ms)]
        batches = [m.avg_per_expert_batch_size for m in mode_metrics.values() if _is_finite(m.avg_per_expert_batch_size)]
        mode_avgs[mode] = {
            "throughput": _mean(tputs),
            "tail_throughput": _mean(tail_tputs),
            "latency": _mean(latencies),
            "batch_size": _mean(batches),
        }

    def _fmt(mode: str, metric: str) -> str:
        if mode not in mode_avgs:
            return "N/A"
        val = mode_avgs[mode].get(metric, float("nan"))
        return f"{val:.3f}" if _is_finite(val) else "N/A"

    print(f"{'Throughput (req/s)':<35} | {_fmt('async', 'throughput'):>15} | "
          f"{_fmt('sync', 'throughput'):>15} | {_fmt('tbo', 'throughput'):>15}")
    print(f"{'Tail Throughput (req/s)':<35} | {_fmt('async', 'tail_throughput'):>15} | "
          f"{_fmt('sync', 'tail_throughput'):>15} | {_fmt('tbo', 'tail_throughput'):>15}")
    print(f"{'Avg Token Latency (ms)':<35} | {_fmt('async', 'latency'):>15} | "
          f"{_fmt('sync', 'latency'):>15} | {_fmt('tbo', 'latency'):>15}")
    print(f"{'Avg Expert Batch Size':<35} | {_fmt('async', 'batch_size'):>15} | "
          f"{_fmt('sync', 'batch_size'):>15} | {_fmt('tbo', 'batch_size'):>15}")
    print("=" * 100 + "\n")


def _empirical_cdf(values: List[float]) -> Tuple[List[float], List[float]]:
    values_sorted = sorted(values)
    n = len(values_sorted)
    ys = [(i + 1) / n for i in range(n)]
    return values_sorted, ys


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / float(len(values)))


def _p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    n = len(values_sorted)
    # Nearest-rank p95 (1-indexed): ceil(0.95 * n)
    rank = int(math.ceil(0.95 * n))
    idx = min(max(rank - 1, 0), n - 1)
    return float(values_sorted[idx])


def _plot_cdf(
    *,
    out_path: str,
    title: str,
    xs: List[float],
    ys: List[float],
    xlabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot empirical CDFs of sampled per-token ITL from experiment logs."
    )
    parser.add_argument(
        "--in-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "per-token-stats"),
        help="Directory containing per_token_stats_*.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "per-token-stats-plots"
        ),
        help="Directory to write PNG plots into (created if needed).",
    )
    parser.add_argument(
        "--unit",
        choices=("ms", "ticks"),
        default="ms",
        help="Which latency column to plot.",
    )
    parser.add_argument(
        "--mode",
        choices=("async", "sync", "tbo"),
        default=None,
        help="Optional filter: only plot one mode.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Skip configs with fewer than this many samples.",
    )
    parser.add_argument(
        "--sim-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing experiment_results_*.txt files for throughput data.",
    )
    args = parser.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    sim_dir = os.path.abspath(args.sim_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load experiment results for throughput/batch size data
    experiment_results = _load_experiment_results(sim_dir)

    latencies_by_cfg: Dict[ConfigKey, Dict[str, List[float]]] = {}
    total_rows = 0
    total_used = 0

    for _, row in _iter_csv_rows(in_dir):
        total_rows += 1
        mode, key = _key_from_row(row)
        if args.mode is not None and mode != args.mode:
            continue
        latency = _extract_latency(row, args.unit)
        if latency is None:
            continue
        total_used += 1
        cfg_bucket = latencies_by_cfg.setdefault(key, {})
        cfg_bucket.setdefault(mode, []).append(latency)

    if not latencies_by_cfg:
        raise SystemExit(
            f"No usable samples found under {in_dir} (rows={total_rows}, used={total_used})."
        )

    xlabel = "ITL (ms)" if args.unit == "ms" else "ITL (ticks)"
    written = 0
    skipped = 0

    def _sort_float(x: float | None) -> tuple[int, float]:
        if x is None:
            return (1, 0.0)
        return (0, float(x))

    for key in sorted(
        latencies_by_cfg.keys(),
        key=lambda k: (
            k.ep_group_size,
            k.global_request_max_batch_size,
            k.attn_service_t,
            k.n_gpu_per_host,
            _sort_float(k.net_delay_intra_host),
            _sort_float(k.net_delay_inter_host),
            k.hidden_dim,
            k.inter_node_bw_gbps,
            0 if k.bw_aware else 1,
        ),
    ):
        by_mode = latencies_by_cfg[key]
        async_values = by_mode.get("async") or []
        sync_values = by_mode.get("sync") or []
        tbo_values = by_mode.get("tbo") or []

        if args.mode is not None:
            if args.mode == "async":
                values = async_values
            elif args.mode == "sync":
                values = sync_values
            else:
                values = tbo_values
            if len(values) < int(args.min_samples):
                skipped += 1
                continue
            series: List[tuple[str, List[float], List[float]]] = []
            xs, ys = _empirical_cdf(values)
            series.append((args.mode, xs, ys))
            mode_desc = args.mode
        else:
            # Compare available modes on the same plot when they are "valid".
            have_async = len(async_values) >= int(args.min_samples)
            have_sync = len(sync_values) >= int(args.min_samples)
            have_tbo = len(tbo_values) >= int(args.min_samples)

            if not have_async and not have_sync and not have_tbo:
                skipped += 1
                continue

            series = []
            mode_desc_parts: List[str] = []
            if have_async:
                xs, ys = _empirical_cdf(async_values)
                series.append(("async", xs, ys))
                mode_desc_parts.append(f"async(n={len(async_values)})")
            if have_sync:
                xs, ys = _empirical_cdf(sync_values)
                series.append(("sync", xs, ys))
                mode_desc_parts.append(f"sync(n={len(sync_values)})")
            if have_tbo:
                xs, ys = _empirical_cdf(tbo_values)
                series.append(("tbo", xs, ys))
                mode_desc_parts.append(f"tbo(n={len(tbo_values)})")
            mode_desc = "+".join(mode_desc_parts)

        if key.net_delay_intra_host is None and key.net_delay_inter_host is None:
            net_desc = "net=unknown"
            net_fname = "netunknown"
        elif key.net_delay_intra_host == key.net_delay_inter_host:
            net_desc = f"net={key.net_delay_intra_host:g}"
            net_fname = f"net{_fmt_float_for_filename(float(key.net_delay_intra_host))}"
        else:
            net_desc = (
                f"net_intra={key.net_delay_intra_host:g} "
                f"net_inter={key.net_delay_inter_host:g}"
            )
            net_fname = (
                f"neti{_fmt_float_for_filename(float(key.net_delay_intra_host))}"
                f"_nete{_fmt_float_for_filename(float(key.net_delay_inter_host))}"
            )

        # Hidden dimension and inter-node bandwidth for filename
        hidden_desc = f"hidden={key.hidden_dim}" if key.hidden_dim else ""
        bw_desc = f"bw={key.inter_node_bw_gbps:g}Gbps" if key.inter_node_bw_gbps else ""
        bwa_desc = "bw_aware" if key.bw_aware else "fixed_delay"

        title = (
            f"Sampled ITL CDF ({mode_desc})\n"
            f"ep={key.ep_group_size} "
            f"global_bs={key.global_request_max_batch_size} "
            f"attn_t={key.attn_service_t:g} "
            f"n_gpu_per_host={key.n_gpu_per_host} {net_desc} "
            f"{hidden_desc} {bw_desc} {bwa_desc}"
        )

        filename = (
            f"itl_cdf_{'compare' if args.mode is None else args.mode}"
            f"_ep{key.ep_group_size}"
            f"_gb{key.global_request_max_batch_size}"
            f"_attn{_fmt_float_for_filename(key.attn_service_t)}"
            f"_ngph{key.n_gpu_per_host}"
            f"_{net_fname}"
            f"_hid{key.hidden_dim}"
            f"_bw{int(key.inter_node_bw_gbps)}"
            f"_bwa{1 if key.bw_aware else 0}"
            f"_na{len(async_values)}"
            f"_ns{len(sync_values)}"
            f"_nt{len(tbo_values)}.png"
        )
        out_path = os.path.join(out_dir, filename)

        # Plot (possibly multiple) curves on the same figure.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        for mode, xs, ys in series:
            label = mode
            if mode == "async":
                label = f"async (n={len(async_values)})"
            elif mode == "sync":
                label = f"sync (n={len(sync_values)})"
            elif mode == "tbo":
                label = f"tbo (n={len(tbo_values)})"
            ax.plot(xs, ys, linewidth=1.5, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        if len(series) > 1:
            ax.legend(loc="lower right", frameon=True, fontsize=9)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        written += 1

    # Summary plot: average ITL per config, comparing async vs sync side-by-side.
    summary_items: List[
        tuple[
            str,
            float | None,
            float | None,
            int,
            float | None,
            float | None,
            int,
            float | None,
            float | None,
            int,
        ]
    ] = []
    for key in sorted(
        latencies_by_cfg.keys(),
        key=lambda k: (
            k.ep_group_size,
            k.global_request_max_batch_size,
            k.attn_service_t,
            k.n_gpu_per_host,
            _sort_float(k.net_delay_intra_host),
            _sort_float(k.net_delay_inter_host),
            k.hidden_dim,
            k.inter_node_bw_gbps,
            0 if k.bw_aware else 1,
        ),
    ):
        by_mode = latencies_by_cfg[key]
        async_values = by_mode.get("async") or []
        sync_values = by_mode.get("sync") or []
        tbo_values = by_mode.get("tbo") or []

        async_ok = len(async_values) >= int(args.min_samples)
        sync_ok = len(sync_values) >= int(args.min_samples)
        tbo_ok = len(tbo_values) >= int(args.min_samples)
        if not async_ok and not sync_ok and not tbo_ok:
            continue

        if key.net_delay_intra_host is None and key.net_delay_inter_host is None:
            net_label = "net=?"
        elif key.net_delay_intra_host == key.net_delay_inter_host:
            net_label = f"net={key.net_delay_intra_host:g}"
        else:
            net_label = f"neti={key.net_delay_intra_host:g}/nete={key.net_delay_inter_host:g}"

        label = (
            f"ep{key.ep_group_size} "
            f"gb{key.global_request_max_batch_size} "
            f"attn{key.attn_service_t:g} "
            f"hid{key.hidden_dim} "
            f"bw{int(key.inter_node_bw_gbps)}"
            f"{' bwa' if key.bw_aware else ' nobw'}"
        )
        summary_items.append(
            (
                label,
                (_mean(async_values) if async_ok else None),
                (_p95(async_values) if async_ok else None),
                len(async_values),
                (_mean(sync_values) if sync_ok else None),
                (_p95(sync_values) if sync_ok else None),
                len(sync_values),
                (_mean(tbo_values) if tbo_ok else None),
                (_p95(tbo_values) if tbo_ok else None),
                len(tbo_values),
            )
        )

    if summary_items and args.mode is None:
        labels = [it[0] for it in summary_items]
        async_means = [it[1] for it in summary_items]
        async_p95s = [it[2] for it in summary_items]
        sync_means = [it[4] for it in summary_items]
        sync_p95s = [it[5] for it in summary_items]
        tbo_means = [it[7] for it in summary_items]
        tbo_p95s = [it[8] for it in summary_items]

        # Collect throughput data from experiment_results
        async_tputs: List[float | None] = []
        sync_tputs: List[float | None] = []
        tbo_tputs: List[float | None] = []

        # Re-iterate to get keys in same order
        sorted_keys = sorted(
            latencies_by_cfg.keys(),
            key=lambda k: (
                k.ep_group_size,
                k.global_request_max_batch_size,
                k.attn_service_t,
                k.n_gpu_per_host,
                _sort_float(k.net_delay_intra_host),
                _sort_float(k.net_delay_inter_host),
                k.hidden_dim,
                k.inter_node_bw_gbps,
                0 if k.bw_aware else 1,
            ),
        )
        for key in sorted_keys:
            by_mode = latencies_by_cfg[key]
            async_values = by_mode.get("async") or []
            sync_values = by_mode.get("sync") or []
            tbo_values = by_mode.get("tbo") or []

            async_ok = len(async_values) >= int(args.min_samples)
            sync_ok = len(sync_values) >= int(args.min_samples)
            tbo_ok = len(tbo_values) >= int(args.min_samples)
            if not async_ok and not sync_ok and not tbo_ok:
                continue

            # Get throughput from experiment results
            async_metrics = experiment_results.get("async", {}).get(key)
            sync_metrics = experiment_results.get("sync", {}).get(key)
            tbo_metrics = experiment_results.get("tbo", {}).get(key)

            async_tputs.append(
                async_metrics.avg_throughput_req_per_sec
                if async_metrics and _is_finite(async_metrics.avg_throughput_req_per_sec)
                else None
            )
            sync_tputs.append(
                sync_metrics.avg_throughput_req_per_sec
                if sync_metrics and _is_finite(sync_metrics.avg_throughput_req_per_sec)
                else None
            )
            tbo_tputs.append(
                tbo_metrics.avg_throughput_req_per_sec
                if tbo_metrics and _is_finite(tbo_metrics.avg_throughput_req_per_sec)
                else None
            )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cfg = len(summary_items)
        fig_w = max(10.0, min(26.0, 1.2 * n_cfg))

        # Create 3-row figure: throughput, avg ITL, p95 ITL
        fig, (ax_tput, ax_avg, ax_p95) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(fig_w, 11.0),
            dpi=160,
            sharex=True,
        )

        xs = list(range(n_cfg))
        modes = ["async", "sync", "tbo"]
        width = 0.8 / float(len(modes))

        def _to_num(vals: List[float | None]) -> List[float]:
            return [float("nan") if v is None else float(v) for v in vals]

        # Throughput plot
        ax_tput.bar(
            [x - width for x in xs],
            _to_num(async_tputs),
            width=width,
            label="async",
            color="C0",
        )
        ax_tput.bar(
            xs,
            _to_num(sync_tputs),
            width=width,
            label="sync",
            color="C1",
        )
        ax_tput.bar(
            [x + width for x in xs],
            _to_num(tbo_tputs),
            width=width,
            label="tbo",
            color="C2",
        )
        ax_tput.set_ylabel("Throughput (req/s)")
        ax_tput.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_tput.legend(loc="best", frameon=True, fontsize=9)

        # Avg ITL plot
        ax_avg.bar(
            [x - width for x in xs],
            _to_num(async_means),
            width=width,
            label="async",
            color="C0",
        )
        ax_avg.bar(
            xs,
            _to_num(sync_means),
            width=width,
            label="sync",
            color="C1",
        )
        ax_avg.bar(
            [x + width for x in xs],
            _to_num(tbo_means),
            width=width,
            label="tbo",
            color="C2",
        )

        ax_avg.set_ylabel(xlabel.replace("ITL", "Avg ITL"))
        ax_avg.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_avg.legend(loc="best", frameon=True, fontsize=9)

        # P95 ITL plot
        ax_p95.bar(
            [x - width for x in xs],
            _to_num(async_p95s),
            width=width,
            label="async",
            color="C0",
        )
        ax_p95.bar(
            xs,
            _to_num(sync_p95s),
            width=width,
            label="sync",
            color="C1",
        )
        ax_p95.bar(
            [x + width for x in xs],
            _to_num(tbo_p95s),
            width=width,
            label="tbo",
            color="C2",
        )

        fig.suptitle("Simulation Summary: Throughput + ITL by Config", y=0.98, fontsize=12)

        ax_p95.set_ylabel(xlabel.replace("ITL", "P95 ITL"))
        ax_p95.set_xlabel("Config")
        ax_p95.set_xticks(xs)
        ax_p95.set_xticklabels(labels, rotation=35, ha="right")
        ax_p95.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_p95.legend(loc="best", frameon=True, fontsize=9)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"itl_summary_compare_{args.unit}.png")
        fig.savefig(out_path)
        plt.close(fig)

    # Print stats summary to stdout
    _print_stats_summary(latencies_by_cfg, experiment_results, args.min_samples, args.unit)

    print(
        f"Read {total_rows} rows, used {total_used} samples from {in_dir}.\n"
        f"Wrote {written} PNG(s) to {out_dir} (skipped {skipped} config(s) with <{args.min_samples} samples)."
    )


if __name__ == "__main__":
    main()
