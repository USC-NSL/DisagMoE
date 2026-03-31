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
                tput = float(m.group(1)) * 1000  # k -> raw
                if tput < 1:  # skip warmup lines with ~0 tput
                    continue
                samples.append(Sample(
                    tput=tput,
                    itl_mean=float(m.group(2)),
                    itl_median=float(m.group(3)),
                    itl_p99=float(m.group(4)),
                ))
    return samples


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
                if tput < 1:  # skip warmup
                    continue
                samples.append(Sample(
                    tput=tput,
                    itl_mean=float(m.group(2)),
                    itl_median=float(m.group(3)),
                    itl_p99=float(m.group(4)),
                ))
    return samples


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

    peak_tput = max(s.tput for s in samples)
    avg_tput = sum(s.tput for s in samples) / len(samples)

    w_itl_mean = sum(s.tput * s.itl_mean for s in samples) / total_weight
    w_itl_median = sum(s.tput * s.itl_median for s in samples) / total_weight
    w_itl_p99 = sum(s.tput * s.itl_p99 for s in samples) / total_weight

    return {
        "num_samples": len(samples),
        "peak_tput": round(peak_tput, 1),
        "avg_tput": round(avg_tput, 1),
        "weighted_itl_mean_ms": round(w_itl_mean, 2),
        "weighted_itl_median_ms": round(w_itl_median, 2),
        "weighted_itl_p99_ms": round(w_itl_p99, 2),
    }


def find_log(run_dir: str, system: str) -> str | None:
    """Find the server log file for a given run directory."""
    if system == "asyncmoe":
        p = os.path.join(run_dir, "server.log")
        return p if os.path.isfile(p) else None
    else:  # sglang
        p = os.path.join(run_dir, "logs", "server_head.log")
        return p if os.path.isfile(p) else None


def parse_run_name(dirname: str):
    """Extract system, workload, rate from directory name.

    Examples:
        asyncmoe-sharegpt_balanced-100rps -> (asyncmoe, sharegpt_balanced, 100)
        sglang_ep16-gsm8k_balanced-200rps -> (sglang_ep16, gsm8k_balanced, 200)
    """
    m = re.match(r'^(asyncmoe|sglang_ep16)-(sharegpt_balanced|gsm8k_balanced)-(\d+)rps$', dirname)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", help="Root experiment directory")
    parser.add_argument("--csv", default=None, help="Output CSV path (default: <experiment_dir>/metrics.csv)")
    args = parser.parse_args()

    exp_dir = args.experiment_dir
    csv_path = args.csv or os.path.join(exp_dir, "metrics.csv")

    rows = []

    # AsyncMoE
    amoe_dir = os.path.join(exp_dir, "asyncmoe-gptoss-results")
    if os.path.isdir(amoe_dir):
        for d in sorted(os.listdir(amoe_dir)):
            system, workload, rate = parse_run_name(d)
            if system is None:
                continue
            log_path = find_log(os.path.join(amoe_dir, d), "asyncmoe")
            if not log_path:
                print(f"WARN: no server.log for {d}", file=sys.stderr)
                continue
            samples = parse_asyncmoe_log(log_path)
            metrics = compute_weighted_metrics(samples)
            if metrics:
                rows.append({"system": system, "workload": workload, "rate_rps": rate, **metrics})
                print(f"  {d}: {metrics['num_samples']} samples, peak_tput={metrics['peak_tput']}, "
                      f"w_itl_mean={metrics['weighted_itl_mean_ms']}ms")
            else:
                print(f"WARN: no usable samples for {d}", file=sys.stderr)

    # SGLang
    sg_dir = os.path.join(exp_dir, "sglang-gptoss-results")
    if os.path.isdir(sg_dir):
        for d in sorted(os.listdir(sg_dir)):
            system, workload, rate = parse_run_name(d)
            if system is None:
                continue
            log_path = find_log(os.path.join(sg_dir, d), "sglang")
            if not log_path:
                print(f"WARN: no server_head.log for {d}", file=sys.stderr)
                continue
            samples = parse_sglang_log(log_path)
            metrics = compute_weighted_metrics(samples)
            if metrics:
                rows.append({"system": system, "workload": workload, "rate_rps": rate, **metrics})
                print(f"  {d}: {metrics['num_samples']} samples, peak_tput={metrics['peak_tput']}, "
                      f"w_itl_mean={metrics['weighted_itl_mean_ms']}ms")
            else:
                print(f"WARN: no usable samples for {d}", file=sys.stderr)

    if not rows:
        print("ERROR: no data found", file=sys.stderr)
        sys.exit(1)

    # Sort: system, workload, rate
    rows.sort(key=lambda r: (r["system"], r["workload"], r["rate_rps"]))

    # Write CSV
    fieldnames = ["system", "workload", "rate_rps", "num_samples", "peak_tput", "avg_tput",
                  "weighted_itl_mean_ms", "weighted_itl_median_ms", "weighted_itl_p99_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
