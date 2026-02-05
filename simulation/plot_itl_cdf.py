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

    return mode, ConfigKey(
        ep_group_size=_safe_int(row.get("ep_group_size"), 0),
        global_request_max_batch_size=_safe_int(
            row.get("global_request_max_batch_size"), 0
        ),
        attn_service_t=float(attn_service_t),
        n_gpu_per_host=int(n_gpu_per_host),
        net_delay_intra_host=intra_opt,
        net_delay_inter_host=inter_opt,
    )


def _extract_latency(row: Dict[str, str], unit: str) -> float | None:
    if unit == "ms":
        val = _safe_float(row.get("latency_ms"), float("nan"))
    elif unit == "ticks":
        val = _safe_float(row.get("latency_ticks"), float("nan"))
    else:
        raise ValueError("unit must be 'ms' or 'ticks'")
    return val if _is_finite(val) else None


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
    args = parser.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

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

        title = (
            f"Sampled ITL CDF ({mode_desc})\n"
            f"ep={key.ep_group_size} "
            f"global_bs={key.global_request_max_batch_size} "
            f"attn_t={key.attn_service_t:g} "
            f"n_gpu_per_host={key.n_gpu_per_host} {net_desc} "
        )

        filename = (
            f"itl_cdf_{'compare' if args.mode is None else args.mode}"
            f"_ep{key.ep_group_size}"
            f"_gb{key.global_request_max_batch_size}"
            f"_attn{_fmt_float_for_filename(key.attn_service_t)}"
            f"_ngph{key.n_gpu_per_host}"
            f"_{net_fname}"
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
            f"ngph{key.n_gpu_per_host} "
            f"{net_label}"
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

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cfg = len(summary_items)
        fig_w = max(10.0, min(26.0, 1.2 * n_cfg))
        fig, (ax_avg, ax_p95) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(fig_w, 8.0),
            dpi=160,
            sharex=True,
        )

        xs = list(range(n_cfg))
        modes = ["async", "sync", "tbo"]
        width = 0.8 / float(len(modes))

        def _to_num(vals: List[float | None]) -> List[float]:
            return [float("nan") if v is None else float(v) for v in vals]

        ax_avg.bar(
            [x - width for x in xs],
            _to_num(async_means),
            width=width,
            label="async (avg)",
        )
        ax_avg.bar(
            xs,
            _to_num(sync_means),
            width=width,
            label="sync (avg)",
        )
        ax_avg.bar(
            [x + width for x in xs],
            _to_num(tbo_means),
            width=width,
            label="tbo (avg)",
        )

        ax_p95.bar(
            [x - width for x in xs],
            _to_num(async_p95s),
            width=width,
            label="async (p95)",
        )
        ax_p95.bar(
            xs,
            _to_num(sync_p95s),
            width=width,
            label="sync (p95)",
        )
        ax_p95.bar(
            [x + width for x in xs],
            _to_num(tbo_p95s),
            width=width,
            label="tbo (p95)",
        )

        fig.suptitle("Sampled ITL summary by config (avg + p95)", y=0.98)

        ax_avg.set_ylabel(xlabel.replace("ITL", "Avg ITL"))
        ax_avg.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_avg.legend(loc="best", frameon=True, fontsize=9)

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

    print(
        f"Read {total_rows} rows, used {total_used} samples from {in_dir}.\n"
        f"Wrote {written} PNG(s) to {out_dir} (skipped {skipped} config(s) with <{args.min_samples} samples)."
    )


if __name__ == "__main__":
    main()
