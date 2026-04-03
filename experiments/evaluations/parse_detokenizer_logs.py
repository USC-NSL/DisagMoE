#!/usr/bin/env python3
"""Parse detokenizer throughput/ITL logs from AsyncMoE and SGLang server logs.

For each log file, extracts per-second throughput and ITL metrics, then computes
throughput-weighted overall ITL statistics. The weighting ensures that high-throughput
periods (steady state) dominate the aggregate, rather than ramp-up/drain periods.

Usage:
    python parse_detokenizer_logs.py <experiment_dir> [--csv <output.csv>]

The script auto-discovers asyncmoe-gptoss-results/ and sglang-gptoss-results/ subdirs.
"""
import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    tput: float          # tokens/s
    itl_mean: float      # ms
    itl_median: float    # ms (p50)
    itl_p99: float       # ms
    timestamp: float = 0.0  # epoch seconds (0 if unavailable)


def _parse_asyncmoe_timestamp(line: str) -> float:
    m = re.search(r'(\d+\.\d{4})\s*-\s*\[INFO\]', line)
    return float(m.group(1)) if m else 0.0


def parse_asyncmoe_log(log_path: str) -> list[Sample]:
    """Parse AsyncMoE detokenizer log lines.

    Format: "Detokenizer: token throughput: 13.95k tokens/s | ITL mean=150.1ms p50=150.0ms p99=160.0ms"
    """
    pattern = re.compile(
        r'token throughput:\s*([\d.]+)k\s*tokens/s'
        r'\s*\|\s*ITL\s+mean=([\d.]+)ms'
        r'\s+p50=([\d.]+)ms'
        r'\s+p99=([\d.]+)ms'
    )
    samples = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                tput = float(m.group(1)) * 1000
                if tput < 1:
                    continue
                samples.append(Sample(
                    tput=tput,
                    itl_mean=float(m.group(2)),
                    itl_median=float(m.group(3)),
                    itl_p99=float(m.group(4)),
                    timestamp=_parse_asyncmoe_timestamp(line),
                ))
    return samples


def _parse_sglang_timestamp(line: str) -> float:
    from datetime import datetime
    m = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return 0.0


def parse_sglang_log(log_path: str) -> list[Sample]:
    """Parse SGLang detokenizer log lines.

    Format: "[...] from Detokenizer Manager, Throughput: 12019.2 tokens/s, In-flight requests: 1818, ...,
             ITL mean=147.34 ms, median=147.59 ms, p99=162.77 ms, samples=118680"
    """
    pattern = re.compile(
        r'Throughput:\s*([\d.]+)\s*tokens/s.*'
        r'ITL\s+mean=([\d.]+)\s*ms.*'
        r'median=([\d.]+)\s*ms.*'
        r'p99=([\d.]+)\s*ms'
    )
    samples = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                tput = float(m.group(1))
                if tput < 1:
                    continue
                samples.append(Sample(
                    tput=tput,
                    itl_mean=float(m.group(2)),
                    itl_median=float(m.group(3)),
                    itl_p99=float(m.group(4)),
                    timestamp=_parse_sglang_timestamp(line),
                ))
    return samples


def max_running_avg(samples: list[Sample], window_sec: float = 60) -> float:
    if not samples:
        return 0.0
    has_ts = samples[0].timestamp > 0
    if has_ts:
        t0 = samples[0].timestamp
        times = [s.timestamp - t0 for s in samples]
    else:
        times = [i * 10.0 for i in range(len(samples))]
    tputs = [s.tput for s in samples]
    n = len(tputs)
    best = 0.0
    for i in range(n):
        total = 0.0
        count = 0
        for j in range(i, n):
            if times[j] - times[i] > window_sec:
                break
            total += tputs[j]
            count += 1
        if count > 0:
            avg = total / count
            if avg > best:
                best = avg
    return best


def compute_weighted_metrics(samples: list[Sample]) -> dict:
    """Compute throughput-weighted aggregate metrics.

    Each sample's ITL values are weighted by its throughput, so steady-state
    (high-tput) periods dominate over ramp-up/drain (low-tput) periods.
    """
    if not samples:
        return {}

    total_weight = sum(s.tput for s in samples)
    if total_weight == 0:
        return {}

    tput_60s = max_running_avg(samples, 60)
    avg_tput = sum(s.tput for s in samples) / len(samples)

    w_itl_mean = sum(s.tput * s.itl_mean for s in samples) / total_weight
    w_itl_median = sum(s.tput * s.itl_median for s in samples) / total_weight
    w_itl_p99 = sum(s.tput * s.itl_p99 for s in samples) / total_weight

    return {
        "num_samples": len(samples),
        "tput_60s": round(tput_60s, 1),
        "avg_tput": round(avg_tput, 1),
        "weighted_itl_mean_ms": round(w_itl_mean, 2),
        "weighted_itl_median_ms": round(w_itl_median, 2),
        "weighted_itl_p99_ms": round(w_itl_p99, 2),
    }


def window_filter(samples: list[Sample], start_s: float, end_s: float) -> list[Sample]:
    if not samples or samples[0].timestamp == 0:
        return samples
    t0 = samples[0].timestamp
    return [s for s in samples if start_s <= (s.timestamp - t0) <= end_s]


def find_log(run_dir: str, system: str) -> str | None:
    """Find the server log file for a given run directory."""
    if system == "asyncmoe":
        p = os.path.join(run_dir, "server.log")
        return p if os.path.isfile(p) else None
    else:  # sglang
        p = os.path.join(run_dir, "logs", "server_head.log")
        return p if os.path.isfile(p) else None


def parse_run_name(dirname: str):
    """Extract system, workload, rate, and optional tag from directory name.

    Supports two naming conventions:
      Old (with rate):  asyncmoe-sharegpt_balanced-100rps[-tag]
      New (no rate):    asyncmoe-sharegpt_regular, sglang_ep16-gsm8k_balanced

    Examples:
        asyncmoe-sharegpt_balanced-100rps -> (asyncmoe, sharegpt_balanced, 100, "")
        sglang_ep16-gsm8k_balanced-200rps -> (sglang_ep16, gsm8k_balanced, 200, "")
        asyncmoe-sharegpt_balanced-400rps-aws-ring -> (asyncmoe, sharegpt_balanced, 400, "aws-ring")
        asyncmoe-sharegpt_regular -> (asyncmoe, sharegpt_regular, 0, "")
        sglang_ep16-gsm8k_balanced -> (sglang_ep16, gsm8k_balanced, 0, "")
        sglang_ep8-sharegpt_regular -> (sglang_ep8, sharegpt_regular, 0, "")
    """
    # Old format: system-workload-NNNrps[-tag]
    m = re.match(r'^(asyncmoe|sglang_\w+)-(sharegpt_(?:balanced|regular)|gsm8k_(?:balanced|regular))-(\d+)rps(?:-(.+))?$', dirname)
    if m:
        tag = m.group(4) or ""
        return m.group(1), m.group(2), int(m.group(3)), tag
    # New format: system-workload[-tag] (no rate)
    m = re.match(r'^(asyncmoe|sglang_\w+)-(sharegpt_(?:balanced|regular)|gsm8k_(?:balanced|regular))(?:-(.+))?$', dirname)
    if m:
        tag = m.group(3) or ""
        return m.group(1), m.group(2), 0, tag
    return None, None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", help="Root experiment directory")
    parser.add_argument("--csv", default=None, help="Output CSV path (default: <experiment_dir>/metrics.csv)")
    parser.add_argument("--window", default=None, help="Time window in seconds as START,END (e.g. 15,60). "
                        "Only samples within this elapsed-time range are included.")
    args = parser.parse_args()

    window_start, window_end = None, None
    if args.window:
        parts = args.window.split(",")
        window_start, window_end = float(parts[0]), float(parts[1])

    exp_dir = args.experiment_dir
    csv_path = args.csv or os.path.join(exp_dir, "metrics.csv")

    rows = []

    result_dirs = []
    for entry in sorted(os.listdir(exp_dir)):
        full = os.path.join(exp_dir, entry)
        if os.path.isdir(full) and entry.endswith("-results"):
            if "asyncmoe" in entry:
                result_dirs.append((full, "asyncmoe", entry))
            elif "sglang" in entry:
                result_dirs.append((full, "sglang", entry))

    for results_path, sys_type, results_name in result_dirs:
        print(f"\nScanning {results_name}/")
        for d in sorted(os.listdir(results_path)):
            run_path = os.path.join(results_path, d)
            if not os.path.isdir(run_path):
                continue
            system, workload, rate, tag = parse_run_name(d)
            if system is None:
                continue
            log_path = find_log(run_path, sys_type)
            if not log_path:
                print(f"WARN: no log for {d}", file=sys.stderr)
                continue
            parse_fn = parse_asyncmoe_log if sys_type == "asyncmoe" else parse_sglang_log
            samples = parse_fn(log_path)
            if window_start is not None:
                samples = window_filter(samples, window_start, window_end)
            metrics = compute_weighted_metrics(samples)
            if metrics:
                result_tag = tag
                if "halfkv" in results_name:
                    result_tag = "halfkv" + (f"-{tag}" if tag else "")
                elif "caprr" in results_name:
                    result_tag = "caprr" + (f"-{tag}" if tag else "")
                rows.append({"system": system, "workload": workload, "rate_rps": rate, "tag": result_tag, **metrics})
                print(f"  {d}: {metrics['num_samples']} samples, tput_60s={metrics['tput_60s']}, "
                      f"w_itl_mean={metrics['weighted_itl_mean_ms']}ms")
            else:
                print(f"WARN: no usable samples for {d}", file=sys.stderr)

    if not rows:
        print("ERROR: no data found", file=sys.stderr)
        sys.exit(1)

    # Sort: system, workload, rate
    rows.sort(key=lambda r: (r["system"], r["workload"], r["rate_rps"], r.get("tag", "")))

    # Write CSV
    fieldnames = ["system", "workload", "rate_rps", "tag", "num_samples", "tput_60s", "avg_tput",
                   "weighted_itl_mean_ms", "weighted_itl_median_ms", "weighted_itl_p99_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
