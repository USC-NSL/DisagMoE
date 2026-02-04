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
    mode: str
    ep_group_size: int
    global_request_max_batch_size: int
    attn_service_t: float
    net_delay: float


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


def _key_from_row(row: Dict[str, str]) -> ConfigKey:
    mode = (row.get("mode") or "").strip() or "unknown"
    return ConfigKey(
        mode=mode,
        ep_group_size=_safe_int(row.get("ep_group_size"), 0),
        global_request_max_batch_size=_safe_int(
            row.get("global_request_max_batch_size"), 0
        ),
        attn_service_t=_safe_float(row.get("attn_service_t"), float("nan")),
        net_delay=_safe_float(row.get("net_delay"), float("nan")),
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
        choices=("async", "sync"),
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

    latencies_by_cfg: Dict[ConfigKey, List[float]] = {}
    total_rows = 0
    total_used = 0

    for _, row in _iter_csv_rows(in_dir):
        total_rows += 1
        key = _key_from_row(row)
        if args.mode is not None and key.mode != args.mode:
            continue
        latency = _extract_latency(row, args.unit)
        if latency is None:
            continue
        total_used += 1
        latencies_by_cfg.setdefault(key, []).append(latency)

    if not latencies_by_cfg:
        raise SystemExit(
            f"No usable samples found under {in_dir} (rows={total_rows}, used={total_used})."
        )

    xlabel = "ITL (ms)" if args.unit == "ms" else "ITL (ticks)"
    written = 0
    skipped = 0

    for key in sorted(
        latencies_by_cfg.keys(),
        key=lambda k: (
            k.mode,
            k.ep_group_size,
            k.global_request_max_batch_size,
            k.attn_service_t,
            k.net_delay,
        ),
    ):
        values = latencies_by_cfg[key]
        if len(values) < int(args.min_samples):
            skipped += 1
            continue
        xs, ys = _empirical_cdf(values)
        title = (
            f"Sampled ITL CDF ({key.mode})\n"
            f"ep={key.ep_group_size} "
            f"global_bs={key.global_request_max_batch_size} "
            f"attn_t={key.attn_service_t:g} net={key.net_delay:g} "
            f"(n={len(values)})"
        )
        filename = (
            f"itl_cdf_{key.mode}"
            f"_ep{key.ep_group_size}"
            f"_gb{key.global_request_max_batch_size}"
            f"_attn{_fmt_float_for_filename(key.attn_service_t)}"
            f"_net{_fmt_float_for_filename(key.net_delay)}"
            f"_n{len(values)}.png"
        )
        out_path = os.path.join(out_dir, filename)
        _plot_cdf(out_path=out_path, title=title, xs=xs, ys=ys, xlabel=xlabel)
        written += 1

    print(
        f"Read {total_rows} rows, used {total_used} samples from {in_dir}.\n"
        f"Wrote {written} PNG(s) to {out_dir} (skipped {skipped} config(s) with <{args.min_samples} samples)."
    )


if __name__ == "__main__":
    main()

