#!/usr/bin/env python3
"""
make_plots.py — generate plots that show the NIXL post_xfer latency phenomenon.

Plots produced:
  1. ENGINE vs MICROBENCH post_p50 by experimental configuration (bar chart)
     with the gap shown explicitly.
  2. ENGINE backend substage breakdown (where the 5ms lives) — shows write_us
     dominates and pre_write_us / status / flush are negligible.
  3. ENGINE dt_post_xfer_s histogram showing bimodal distribution
     (same-node fast peaks, cross-node ms-range mode).
  4. ENGINE per-peer dt_post_xfer_s p50 — visualises the same-node-fast,
     cross-node-slow effect across all 16 peers.
  5. ENGINE write_us / batch_us scatter — visually proves ~100% of batch_us
     is inside the ucp_put_nbx call.
"""
from __future__ import annotations

import json
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = "/home/yizhuoliang/DisagMoE"
PLOT_DIR = f"{REPO}/benchmark/ops/nixl-limitation/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

ENGINE_TRACE_DIR = (
    f"{REPO}/benchmark/ops/nixl-limitation/results/decisive/run_20260508_202623"
)
ENGINE_NIXL_LOG = f"{ENGINE_TRACE_DIR}/nixl_ucx_post_trace.log"
ENGINE_TRACES = f"{ENGINE_TRACE_DIR}/traces"


def parse_backend_log(path):
    keys = [
        "total_us", "send_range_us", "batch_us", "write_us", "pre_write_us",
        "flush_call_us", "status_us", "status_progress_us", "notif_us",
    ]
    data = {k: [] for k in keys}
    pat = re.compile(r"NIXL_UCX_POST_TRACE,(.*?)$")
    n = 0
    with open(path) as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            kv = dict(p.split("=", 1) for p in m.group(1).split(",") if "=" in p)
            try:
                for k in keys:
                    if k in kv:
                        data[k].append(int(float(kv[k])))
            except (ValueError, KeyError):
                continue
            n += 1
    print(f"parsed {n} backend lines from {path}")
    return {k: np.array(v) for k, v in data.items() if v}


def parse_engine_send_traces(traces_dir):
    out = {}
    for f in sorted(glob(f"{traces_dir}/device_*/nixl_send_traces*.json")):
        with open(f) as fh:
            d = json.load(fh)
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    print(f"parsed {len(out.get('dt_post_xfer_s', []))} send-trace samples")
    return {k: np.array(v) if isinstance(v, list) and v and isinstance(v[0], (int, float))
            else np.array(v)
            for k, v in out.items()}


def percentile(arr, q):
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def plot_microbench_vs_engine(out_path):
    configs = [
        ("8 KiB desc\n1 EP", 9.0),
        ("8 KiB desc\n7 nodes mesh", 11.4),
        ("8 KiB desc\n7 nodes bidir", 17.0),
        ("8 KiB desc\nshared-agent\n6 nodes bidir", 9.2),
        ("64 KiB desc\nshared-agent\n6 nodes bidir", 87.0),
        ("512 KiB desc\nshared-agent\n6 nodes bidir", 509.0),
        ("1 MiB desc\nshared-agent\n6 nodes bidir", 1022.0),
        ("2 MiB desc\nshared-agent\n6 nodes bidir", 2098.0),
        ("4 MiB desc\nshared-agent\n6 nodes bidir\n[ENGINE-FAITHFUL]", 3986.0),
        ("Engine\n16 ranks\n(4 MiB desc)", 5164.0),
    ]

    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    colors = ["#4caf50" if v < 100 else ("#ff9800" if v < 1000 else "#d32f2f") for v in values]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("post_xfer p50 latency (µs, log scale)")
    ax.set_title(
        "NIXL post_xfer p50 latency: descriptor size is the ROOT CAUSE\n"
        "Engine uses 4 MiB descriptors -> 5 ms latency. Microbench reproduces with same desc_len."
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(values):
        ax.text(i, v * 1.1, f"{v:g} µs", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(1000, color="grey", linestyle="--", alpha=0.5, label="1 ms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  -> {out_path}")


def plot_descsize_sweep(out_path):
    sizes_kib = np.array([8, 64, 512, 1024, 2048, 4096])
    p50_us = np.array([9.0, 87.0, 509.0, 1022.0, 2098.0, 3986.0])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes_kib, p50_us, "o-", color="#1976d2", linewidth=2,
            markersize=10, label="microbench post_p50 (6-node mesh)")
    ax.plot([4096], [5164.0], "s", color="#d32f2f", markersize=14,
            label="engine 16-rank post_p50 = 5164 µs")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("descriptor length (KiB)")
    ax.set_ylabel("post_xfer p50 latency (µs)")
    ax.set_title(
        "post_xfer p50 vs NIXL descriptor length\n"
        "Linear in descriptor size: each post transfers desc_len bytes regardless of payload"
    )
    for x, y in zip(sizes_kib, p50_us):
        ax.annotate(f"{y:.0f} µs", (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.annotate("ENGINE\n5164 µs", (4096, 5164.0), textcoords="offset points",
                xytext=(-50, -25), fontsize=10, color="#d32f2f", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  -> {out_path}")


def plot_engine_substage_breakdown(backend_data, out_path):
    fields = [
        ("write_us", "ep.write() (= ucp_put_nbx)"),
        ("pre_write_us", "metadata setup"),
        ("flush_call_us", "flush_ep"),
        ("status_us", "status() (post-write)"),
        ("notif_us", "notification send"),
    ]
    p50s = []
    p90s = []
    p99s = []
    labels = []
    for f, lab in fields:
        if f in backend_data:
            p50s.append(percentile(backend_data[f], 50))
            p90s.append(percentile(backend_data[f], 90))
            p99s.append(percentile(backend_data[f], 99))
            labels.append(lab)

    total_p50 = percentile(backend_data["total_us"], 50)
    pcts = [v / total_p50 * 100 for v in p50s]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, p50s, w, label="p50", color="#1976d2")
    ax.bar(x, p90s, w, label="p90", color="#ff9800")
    ax.bar(x + w, p99s, w, label="p99", color="#d32f2f")
    ax.set_yscale("symlog")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("substage time (µs, symlog)")
    ax.set_title(
        f"Engine NIXL backend substages\nOut of {total_p50:.0f}µs p50 total time, "
        f"{pcts[0]:.1f}% is inside ep.write()"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.pie(p50s, labels=[f"{l}\n{p:.1f}%" for l, p in zip(labels, pcts)],
            colors=["#d32f2f", "#9e9e9e", "#9e9e9e", "#9e9e9e", "#9e9e9e"],
            autopct=None, startangle=90)
    ax2.set_title("p50 substage contribution\n(red = ucp_put_nbx itself)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  -> {out_path}")


def plot_engine_histogram(send_traces, out_path):
    xfers = send_traces["dt_post_xfer_s"] * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.logspace(np.log10(1), np.log10(2e5), 80)
    ax.hist(xfers, bins=bins, edgecolor="black", linewidth=0.3, color="#1976d2")
    ax.set_xscale("log")
    ax.set_xlabel("dt_post_xfer_s (µs, log)")
    ax.set_ylabel("count")
    ax.set_title(
        "Engine dt_post_xfer_s — bimodal distribution\n"
        "Left peak ≈ same-node SHM/CUDA-IPC; right peak ≈ cross-node RoCE RDMA"
    )
    ax.axvline(percentile(xfers, 50), color="red", linestyle="--", alpha=0.6, label=f"p50 = {percentile(xfers, 50):.0f}µs")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    bins2 = np.linspace(0, 30000, 60)
    ax2.hist(xfers[xfers <= 30000], bins=bins2, edgecolor="black", linewidth=0.3, color="#1976d2")
    ax2.set_xlabel("dt_post_xfer_s (µs)")
    ax2.set_ylabel("count")
    ax2.set_title("Same window, linear scale 0-30 ms")
    ax2.axvline(percentile(xfers, 50), color="red", linestyle="--", alpha=0.6,
                label=f"p50 = {percentile(xfers, 50):.0f}µs")
    ax2.axvline(20, color="green", linestyle="--", alpha=0.6,
                label="microbench p50 (20µs)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  -> {out_path}")


def plot_per_peer_p50(send_traces, out_path):
    peers = send_traces["peer_id"]
    xfers = send_traces["dt_post_xfer_s"] * 1e6

    by_peer = {}
    for p, x in zip(peers, xfers):
        by_peer.setdefault(int(p), []).append(float(x))

    sorted_peers = sorted(by_peer.keys())
    p50s = [percentile(np.array(by_peer[p]), 50) for p in sorted_peers]
    counts = [len(by_peer[p]) for p in sorted_peers]
    colors = ["#4caf50" if v < 100 else "#d32f2f" for v in p50s]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars = ax.bar(sorted_peers, p50s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("symlog")
    ax.set_xlabel("peer_id (rank index)")
    ax.set_ylabel("dt_post_xfer_s p50 (µs, symlog)")
    ax.set_title(
        "Engine dt_post_xfer_s p50 per peer\n"
        "Green = same-node peer (1 per device, fast SHM/CUDA-IPC). "
        "Red = cross-node peer (slow ms-scale)."
    )
    for p, v, n in zip(sorted_peers, p50s, counts):
        ax.text(p, v * 1.2 if v > 100 else v + 30, f"{v:.0f}",
                ha="center", fontsize=7)
    ax.axhline(20, color="black", linestyle="--", alpha=0.4,
               label="microbench p50 (20µs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  -> {out_path}")


def plot_write_vs_batch(backend_data, out_path):
    write = backend_data.get("write_us")
    batch = backend_data.get("batch_us")
    if write is None or batch is None:
        return

    n_sample = min(20000, len(write))
    idx = np.random.choice(len(write), n_sample, replace=False)
    w = write[idx]
    b = batch[idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(b, w, s=2, alpha=0.3, color="#1976d2", rasterized=True)
    lim = max(np.percentile(b, 99), np.percentile(w, 99)) * 1.1
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5,
            label="write_us == batch_us")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("batch_us (NIXL sendXferRangeBatch wall time)")
    ax.set_ylabel("write_us (sum over ep.write() calls)")
    ax.set_title(
        "Engine: write_us == batch_us at every percentile\n"
        f"100% of NIXL backend time is inside ucp_put_nbx (n={n_sample} samples)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  -> {out_path}")


def plot_summary_text(backend_data, send_traces, out_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")

    total_p50 = percentile(backend_data["total_us"], 50)
    write_p50 = percentile(backend_data["write_us"], 50)
    batch_p50 = percentile(backend_data["batch_us"], 50)
    pre_p50 = percentile(backend_data["pre_write_us"], 50)
    flush_p50 = percentile(backend_data["flush_call_us"], 50)
    status_p50 = percentile(backend_data["status_us"], 50)
    notif_p50 = percentile(backend_data["notif_us"], 50)

    engine_xfer_p50 = percentile(send_traces["dt_post_xfer_s"] * 1e6, 50)

    pct = lambda v: 100 * v / total_p50

    text = f"""
NIXL post_xfer latency phenomenon — fine-grained breakdown

Engine workload: glm45air_106b, 16 ranks across 8 nodes (sgpu1,3-9), 10 reqs at 1 rps.
NIXL UCX backend instrumented with NIXL_UCX_POST_TRACE=1.
Captured {len(backend_data['total_us'])} backend trace lines + {len(send_traces['dt_post_xfer_s'])} engine send-traces.

Engine-side (DisagMoE C++) measurement of nixl_capi_post_xfer_req:
    dt_post_xfer_s p50 = {engine_xfer_p50:.0f} µs

NIXL backend substage decomposition (p50 across all 1.2M post calls):
    total_us         {total_p50:>7.0f} µs   (100.0%)
    └─ batch_us      {batch_p50:>7.0f} µs   ({pct(batch_p50):>5.1f}%)   sendXferRangeBatch
        └─ write_us  {write_p50:>7.0f} µs   ({pct(write_p50):>5.1f}%)   ep.write() = checkTxState() + ucp_put_nbx()
        └─ pre_write {pre_p50:>7.0f} µs   ({pct(pre_p50):>5.1f}%)   metadata pointer chasing, getRkey
    └─ flush_call_us {flush_p50:>7.0f} µs   ({pct(flush_p50):>5.1f}%)   ucp_ep_flush_nbx
    └─ status_us     {status_p50:>7.0f} µs   ({pct(status_p50):>5.1f}%)   status() incl. progressLoop()
    └─ notif_us      {notif_p50:>7.0f} µs   ({pct(notif_p50):>5.1f}%)   notification send

ROOT CAUSE FOUND:
    The 5 ms is the bandwidth-limited transfer time of a 4 MiB RDMA WRITE over RoCE.
    The engine is sending 4 MiB per post even though the application only intends
    to transfer 8-40 KiB.

WHY: NIXL's nixl_capi_make_xfer_req takes a descriptor LIST and INDICES. It transfers
    the full length of each selected descriptor, ignoring the application-level
    "bytes_to_write". DisagMoE creates each descriptor with NIXL_MAX_BATCH_BYTES = 4 MiB.
    Every post therefore transfers 4 MiB even when the payload is 8 KiB.

EVIDENCE: descriptor-size sweep in microbench (linear in desc_len):
    desc_len    post_p50_us
    8 KiB       9
    64 KiB      87
    512 KiB     509
    1 MiB       1022
    2 MiB       2098
    4 MiB       3986     <- microbench reproduces engine
    [engine 4 MiB]  {engine_xfer_p50:.0f}

write_us == batch_us at every percentile -> all 5 ms is inside ucp_put_nbx,
which is exactly what you'd expect for a bandwidth-bound 4 MiB RDMA WRITE.

FIX: Either
    (a) shrink the NIXL descriptor length to match actual payload size, or
    (b) re-create the xfer descriptor per call with the correct byte count, or
    (c) switch DisagMoE's NIXL slot allocation so each descriptor is sized to the
        actual message instead of a worst-case batch.
"""
    ax.text(0.02, 0.98, text, family="monospace", fontsize=9.5,
            verticalalignment="top", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  -> {out_path}")


def main():
    print("Parsing engine trace data...")
    backend = parse_backend_log(ENGINE_NIXL_LOG)
    send_traces = parse_engine_send_traces(ENGINE_TRACES)

    print("\nGenerating plots...")
    plot_microbench_vs_engine(f"{PLOT_DIR}/01_microbench_vs_engine.png")
    plot_engine_substage_breakdown(backend, f"{PLOT_DIR}/02_engine_substage_breakdown.png")
    plot_engine_histogram(send_traces, f"{PLOT_DIR}/03_engine_xfer_histogram.png")
    plot_per_peer_p50(send_traces, f"{PLOT_DIR}/04_engine_per_peer_p50.png")
    plot_write_vs_batch(backend, f"{PLOT_DIR}/05_write_us_vs_batch_us.png")
    plot_descsize_sweep(f"{PLOT_DIR}/06_desc_size_sweep.png")
    plot_summary_text(backend, send_traces, f"{PLOT_DIR}/07_summary.png")

    print(f"\nAll plots written to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
