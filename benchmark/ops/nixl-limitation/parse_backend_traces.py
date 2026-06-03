#!/usr/bin/env python3
"""
parse_backend_traces.py

Parses NIXL_UCX_POST_TRACE lines emitted by the instrumented NIXL UCX backend at
/tmp/nixl-repo/build-trace-wheelucx (env var NIXL_UCX_POST_TRACE=1).

Trace line format (single line, comma-separated key=value):

    NIXL_UCX_POST_TRACE,total_us=104,send_range_us=94,batch_us=83,
    flush_call_us=3,status_us=4,status_progress_us=4,notif_us=0,
    descs=1,batches=1,batch_descs=1,flushes=1,ret=0

This parser also correlates with DisagMoE's per-hop send-side timings
(`dt_post_xfer_s`) drained from advanced trace JSON files
(`*nixl_send_traces*.json` under a traces/ directory) so we can answer the
fundamental question: "Does the engine's 5ms post_xfer_req live INSIDE the NIXL
backend, or ABOVE the backend (in the C-API wrapper / agent lock / scheduler)?"

Usage:
    python parse_backend_traces.py <log_or_dir> [--traces-dir DIR] \\
        [--threshold-us US] [--out OUT.txt]

Inputs:
    <log_or_dir>     A file or directory containing NIXL_UCX_POST_TRACE lines.
                     If a directory, recursively grep all *.log files.
    --traces-dir D   Optional. Directory containing DisagMoE
                     `*nixl_send_traces*.json` files (e.g. .../traces/).
                     Used to compute dt_post_xfer_s p50/p99 for correlation.
    --threshold-us N Pass/fail threshold (default 1000us). Backend total_us p50
                     below this counts as "fast backend".
    --out FILE       Write a structured analysis report to FILE.

Outputs:
    Per-substage percentile table (p50/p90/p99/max).
    Top-2 substages by p50 contribution (where the time lives).
    Engine vs backend correlation if --traces-dir provided.
    Verdict: ABOVE_BACKEND / INSIDE_BACKEND_FLUSH / INSIDE_BACKEND_STATUS /
             INSIDE_BACKEND_BATCH / NO_GAP / INSUFFICIENT_DATA
    Exit codes:
        0 - parse + analysis succeeded
        2 - no trace lines found
        3 - parse error
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# Field order matches the order emitted by ucx_backend.cpp instrumentation.
BACKEND_FIELDS = [
    "total_us",
    "send_range_us",
    "batch_us",
    "flush_call_us",
    "status_us",
    "status_progress_us",
    "notif_us",
    "write_us",
    "pre_write_us",
    "writes",
    "descs",
    "batches",
    "batch_descs",
    "flushes",
    "ret",
]

# Substages that contribute time (the rest are counters/return codes).
SUBSTAGES = [
    "send_range_us",
    "batch_us",
    "write_us",
    "pre_write_us",
    "flush_call_us",
    "status_us",
    "status_progress_us",
    "notif_us",
]

TIME_FIELDS = ["total_us"] + SUBSTAGES
COUNTER_FIELDS = ["writes", "descs", "batches", "batch_descs", "flushes", "ret"]

TRACE_KEY = "NIXL_UCX_POST_TRACE"
KV_RE = re.compile(r"(\w+)=(-?\d+)")


def _percentile(sorted_xs: List[int], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    pos = (len(sorted_xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def parse_trace_lines(lines: Iterable[str]) -> List[Dict[str, int]]:
    """Parse NIXL_UCX_POST_TRACE lines into list of dicts.

    Tolerates lines with extra prefixes (timestamps, log levels) before the
    NIXL_UCX_POST_TRACE token. Skips lines that don't include the marker.
    """
    rows: List[Dict[str, int]] = []
    for raw in lines:
        if TRACE_KEY not in raw:
            continue
        payload = raw[raw.index(TRACE_KEY):]
        kvs = KV_RE.findall(payload)
        if not kvs:
            continue
        row = {k: int(v) for k, v in kvs}
        if "total_us" not in row:
            continue
        rows.append(row)
    return rows


def collect_lines_from_path(path: str) -> List[str]:
    out: List[str] = []
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(
            glob.glob(os.path.join(path, "**", "*.log"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
        )
    for f in files:
        try:
            with open(f, "r", errors="replace") as fh:
                for line in fh:
                    if TRACE_KEY in line:
                        out.append(line)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: cannot read {f}: {e}", file=sys.stderr)
    return out


def percentile_table(rows: List[Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """Compute p50/p90/p99/max/sum for each TIME_FIELDS field."""
    out: Dict[str, Dict[str, float]] = {}
    for field in TIME_FIELDS:
        vals = sorted(int(r.get(field, 0)) for r in rows)
        out[field] = {
            "p50": _percentile(vals, 0.5),
            "p90": _percentile(vals, 0.9),
            "p99": _percentile(vals, 0.99),
            "max": float(vals[-1]) if vals else 0.0,
            "sum": float(sum(vals)),
            "count": float(len(vals)),
        }
    return out


def counter_summary(rows: List[Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for field in COUNTER_FIELDS:
        vals = [int(r.get(field, 0)) for r in rows]
        if not vals:
            continue
        from collections import Counter
        c = Counter(vals)
        top = c.most_common(3)
        out[field] = {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(sum(vals) / len(vals)),
            "top": top,  # type: ignore[dict-item]
        }
    return out


def load_disagmoe_send_traces(traces_dir: str) -> List[float]:
    """Drain DisagMoE per-hop dt_post_xfer_s values from advanced trace JSONs.

    Looks for files matching *nixl_send_traces*.json under traces_dir.
    Each JSON is a list of records with key "dt_post_xfer_s" (seconds, float).

    Returns a flat list of dt_post_xfer_s values in MICROSECONDS (converted).
    """
    vals: List[float] = []
    paths = glob.glob(os.path.join(traces_dir, "**", "*nixl_send_traces*.json"), recursive=True)
    for p in paths:
        try:
            with open(p, "r") as fh:
                data = json.load(fh)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: cannot read {p}: {e}", file=sys.stderr)
            continue
        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            dt = rec.get("dt_post_xfer_s")
            if dt is None:
                continue
            try:
                vals.append(float(dt) * 1e6)
            except (TypeError, ValueError):
                continue
    return vals


def categorize(stats: Dict[str, Dict[str, float]], engine_p50: Optional[float]) -> str:
    """Return a short verdict label based on dominant substage / above-vs-inside."""
    backend_p50 = stats["total_us"]["p50"]

    # Find dominant substage by p50 (excluding total).
    contribs = [(k, stats[k]["p50"]) for k in SUBSTAGES if k != "send_range_us"]
    contribs.sort(key=lambda kv: kv[1], reverse=True)
    dom = contribs[0][0] if contribs else None

    if engine_p50 is None:
        # Just classify backend internal dominance.
        if backend_p50 < 200:
            return f"BACKEND_FAST(dom={dom})"
        return f"BACKEND_DOMINATED(dom={dom},p50={backend_p50:.0f}us)"

    # We have engine numbers. Compare.
    if engine_p50 < 1500 and backend_p50 < 1500:
        return "NO_GAP"
    ratio = engine_p50 / max(backend_p50, 1.0)
    if backend_p50 < 500 and engine_p50 > 1500:
        # Backend is fast but engine is slow -> ABOVE_BACKEND.
        return f"ABOVE_BACKEND(engine={engine_p50:.0f}us,backend={backend_p50:.0f}us,ratio={ratio:.1f}x)"
    if backend_p50 >= 1500:
        # Backend itself is slow -> INSIDE_BACKEND, dominant substage tells which.
        if dom == "flush_call_us":
            return f"INSIDE_BACKEND_FLUSH(p50={backend_p50:.0f}us)"
        if dom == "status_progress_us" or dom == "status_us":
            return f"INSIDE_BACKEND_STATUS(p50={backend_p50:.0f}us)"
        if dom == "batch_us":
            return f"INSIDE_BACKEND_BATCH(p50={backend_p50:.0f}us)"
        return f"INSIDE_BACKEND_OTHER(p50={backend_p50:.0f}us,dom={dom})"
    return f"MIXED(engine={engine_p50:.0f}us,backend={backend_p50:.0f}us,dom={dom})"


def render_report(
    rows: List[Dict[str, int]],
    stats: Dict[str, Dict[str, float]],
    counters: Dict[str, Dict[str, float]],
    engine_us: List[float],
    threshold_us: float,
) -> str:
    out: List[str] = []
    out.append("=" * 72)
    out.append("NIXL_UCX_POST_TRACE Backend Substage Analysis")
    out.append("=" * 72)
    out.append(f"backend trace lines parsed: {len(rows)}")
    if engine_us:
        out.append(f"DisagMoE dt_post_xfer_s samples: {len(engine_us)}")
    out.append("")
    out.append(f"{'field':24s} {'p50_us':>10s} {'p90_us':>10s} {'p99_us':>10s} {'max_us':>10s}")
    out.append("-" * 72)
    for f in TIME_FIELDS:
        s = stats[f]
        out.append(
            f"{f:24s} {s['p50']:>10.1f} {s['p90']:>10.1f} {s['p99']:>10.1f} {s['max']:>10.1f}"
        )
    out.append("")

    contribs = [(k, stats[k]["p50"]) for k in SUBSTAGES if k != "send_range_us"]
    contribs.sort(key=lambda kv: kv[1], reverse=True)
    out.append("Top substages by p50:")
    for k, v in contribs:
        share = (v / max(stats["total_us"]["p50"], 1e-9)) * 100.0
        out.append(f"  {k:24s} {v:>8.1f}us  ({share:5.1f}% of total)")
    out.append("")

    if counters:
        out.append("Counter summary:")
        for k, c in counters.items():
            out.append(
                f"  {k:24s} min={c['min']:.0f} mean={c['mean']:.2f} max={c['max']:.0f}"
                f"  top={c.get('top')}"
            )
        out.append("")

    engine_p50: Optional[float] = None
    if engine_us:
        engine_sorted = sorted(engine_us)
        engine_p50 = _percentile(engine_sorted, 0.5)
        engine_p99 = _percentile(engine_sorted, 0.99)
        engine_max = engine_sorted[-1]
        out.append("DisagMoE engine vs NIXL backend:")
        out.append(
            f"  engine dt_post_xfer_s     p50={engine_p50:>9.1f}us  "
            f"p99={engine_p99:>9.1f}us  max={engine_max:>9.1f}us"
        )
        backend_p50 = stats["total_us"]["p50"]
        backend_p99 = stats["total_us"]["p99"]
        out.append(
            f"  backend total_us          p50={backend_p50:>9.1f}us  "
            f"p99={backend_p99:>9.1f}us"
        )
        gap = engine_p50 - backend_p50
        ratio = engine_p50 / max(backend_p50, 1.0)
        out.append(
            f"  GAP (engine - backend)    p50={gap:>9.1f}us  ratio={ratio:.2f}x"
        )
        out.append("")

    label = categorize(stats, engine_p50)
    out.append(f"Verdict: {label}")
    out.append("")

    pass_fail = "PASS" if stats["total_us"]["p50"] < threshold_us else "FAIL"
    out.append(
        f"Backend p50 vs threshold ({threshold_us:.0f}us): {pass_fail}"
        f" (p50={stats['total_us']['p50']:.0f}us)"
    )
    out.append("=" * 72)
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="Log file or directory containing NIXL_UCX_POST_TRACE lines")
    ap.add_argument("--traces-dir", default=None,
                    help="Optional dir with DisagMoE *nixl_send_traces*.json files")
    ap.add_argument("--threshold-us", type=float, default=1000.0,
                    help="Backend total_us p50 threshold for PASS/FAIL")
    ap.add_argument("--out", default=None, help="Write report to this file in addition to stdout")
    args = ap.parse_args(argv)

    raw_lines = collect_lines_from_path(args.source)
    rows = parse_trace_lines(raw_lines)
    if not rows:
        print(f"ERROR: no NIXL_UCX_POST_TRACE lines found under {args.source}", file=sys.stderr)
        return 2

    stats = percentile_table(rows)
    counters = counter_summary(rows)
    engine_us: List[float] = []
    if args.traces_dir:
        engine_us = load_disagmoe_send_traces(args.traces_dir)

    report = render_report(rows, stats, counters, engine_us, args.threshold_us)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
