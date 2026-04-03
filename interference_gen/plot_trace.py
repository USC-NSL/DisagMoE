#!/usr/bin/env python3
"""Plot raw De Sensi trace BW and the scaled schedule side by side."""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statistics import median

TRACE_MSG_SIZE = 16777216  # 16 MB
TRACE_DURATION_SEC = 3600
TRACE_WARMUP = 20

TRACE_DATASETS = {
    "oracle_hpc": "raw_traces/2022_07_13_13_09_16/ng_netnoise_mpi_bw.out",
    "aws_hpc_metal": "raw_traces/2022_05_15_18_54_34/ng_netnoise_mpi_bw.out",
    "azure_hpc_200g": "raw_traces/2022_03_25_17_01_24/ng_netnoise_mpi_bw.out",
    "deep_est_ib": "raw_traces/2022_05_19_16_35_07/ng_netnoise_mpi_bw.out",
}


def load_trace(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, TRACE_DATASETS[name])

    rtts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    rtts.append(float(parts[1]))
                except ValueError:
                    pass

    cleaned = [rtt for i, rtt in enumerate(rtts) if (i % 1000) >= TRACE_WARMUP]
    bw_bps = np.array([TRACE_MSG_SIZE / (rtt * 1e-6) for rtt in cleaned])
    bw_gbps = bw_bps * 8 / 1e9

    # Time axis
    sample_interval = TRACE_DURATION_SEC / len(bw_gbps)
    times = np.arange(len(bw_gbps)) * sample_interval

    return times, bw_gbps


def plot_raw_trace(name, link_capacity_gbps=200):
    times, bw_gbps = load_trace(name)

    med_bw = np.median(bw_gbps)
    max_bw = np.max(bw_gbps)
    min_bw = np.min(bw_gbps)
    cov = np.std(bw_gbps) / np.mean(bw_gbps)

    # Scaled BW (what the simulator would produce)
    scaled = (bw_gbps / max_bw) * link_capacity_gbps

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

    # --- Row 1: Full 1-hour raw trace + scaled ---
    ax = axes[0, 0]
    ax.plot(times / 60, bw_gbps, linewidth=0.15, color="steelblue", alpha=0.7,
            rasterized=True)
    ax.axhline(y=med_bw, color="red", linestyle="--", linewidth=0.8,
               label=f"Median: {med_bw:.2f} Gbps")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0, top=max_bw * 1.08)
    ax.set_title(f"Raw trace: {name} (1 hour)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    ax.plot(times / 60, scaled, linewidth=0.15, color="darkorange", alpha=0.7,
            rasterized=True)
    ax.axhline(y=np.median(scaled), color="red", linestyle="--", linewidth=0.8,
               label=f"Median: {np.median(scaled):.1f} Gbps")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0, top=link_capacity_gbps * 1.08)
    ax.set_title(f"Scaled to {link_capacity_gbps} Gbps link (1 hour)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- Row 2: First 10 minutes raw + scaled ---
    # mask_10m = times <= 600
    mask_10m = times <= 5 # temporarily 5 secs
    ax = axes[1, 0]
    ax.plot(times[mask_10m], bw_gbps[mask_10m], linewidth=0.3,
            color="steelblue", alpha=0.8, rasterized=True)
    ax.axhline(y=med_bw, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0, top=max_bw * 1.08)
    # ax.set_title(f"Raw trace: first 10 min", fontsize=11)
    ax.set_title(f"Raw trace: first 5 secs", fontsize=11) # temporarily 5 secs

    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    ax.plot(times[mask_10m], scaled[mask_10m], linewidth=0.3,
            color="darkorange", alpha=0.8, rasterized=True)
    ax.axhline(y=np.median(scaled), color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0, top=link_capacity_gbps * 1.08)
    # ax.set_title(f"Scaled: first 10 min", fontsize=11)
    ax.set_title(f"Scaled: first 5 sec", fontsize=11) # temporarily 5 secs
    ax.grid(True, alpha=0.2)

    # --- Row 3: Histogram + Stats ---
    ax = axes[2, 0]
    ax.hist(bw_gbps, bins=200, color="steelblue", alpha=0.7,
            edgecolor="none", density=True)
    ax.axvline(x=med_bw, color="red", linestyle="--", linewidth=1,
               label=f"Median: {med_bw:.2f}")
    ax.axvline(x=min_bw, color="orange", linestyle=":", linewidth=1,
               label=f"Min: {min_bw:.2f}")
    ax.set_xlabel("Bandwidth (Gbps)")
    ax.set_ylabel("Density")
    ax.set_title("BW Distribution (raw trace)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2, 1]
    ax.axis("off")
    sample_interval_ms = TRACE_DURATION_SEC / len(bw_gbps) * 1000
    stats = [
        f"Dataset:      {name}",
        f"Samples:      {len(bw_gbps):,}",
        f"Duration:     {TRACE_DURATION_SEC}s (1 hour)",
        f"Sample interval: {sample_interval_ms:.2f} ms",
        "",
        f"Median BW:    {med_bw:.2f} Gbps",
        f"Max BW:       {max_bw:.2f} Gbps",
        f"Min BW:       {min_bw:.2f} Gbps",
        f"CoV:          {cov:.6f}",
        "",
        f"--- Scaling to {link_capacity_gbps} Gbps ---",
        f"Reference:    max ({max_bw:.2f} Gbps)",
        f"Median → {np.median(scaled):.1f} Gbps",
        f"Min    → {np.min(scaled):.1f} Gbps",
        f"Max    → {link_capacity_gbps:.1f} Gbps",
    ]
    ax.text(0.05, 0.95, "\n".join(stats), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Trace Statistics", fontsize=11)

    fig.suptitle(f"De Sensi et al. Trace: {name}", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"trace_{name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_receiver_csv(csv_path, trace_name=None, link_capacity_gbps=200):
    """Plot receiver CSV alongside raw trace for comparison."""
    import csv as csv_mod

    recv_times, recv_bw = [], []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            recv_times.append(float(row["window_ms"]) / 1000.0)  # ms → sec
            recv_bw.append(float(row["bw_gbps"]))

    recv_times = np.array(recv_times)
    recv_bw = np.array(recv_bw)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # If trace available, load it for comparison
    if trace_name:
        t_times, t_bw = load_trace(trace_name)
        t_max = np.max(t_bw)
        t_scaled = (t_bw / t_max) * link_capacity_gbps

        # Row 1: Full timeline comparison
        ax = axes[0, 0]
        ax.plot(t_times / 60, t_scaled, linewidth=0.15, color="darkorange",
                alpha=0.6, label="Target (scaled trace)", rasterized=True)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Bandwidth (Gbps)")
        ax.set_ylim(bottom=0, top=link_capacity_gbps * 1.08)
        ax.set_title("Target BW from trace (scaled)", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    else:
        axes[0, 0].text(0.5, 0.5, "No trace for comparison",
                        transform=axes[0, 0].transAxes, ha="center")

    # Receiver full timeline
    ax = axes[0, 1]
    ax.plot(recv_times / 60, recv_bw, linewidth=0.15, color="steelblue",
            alpha=0.7, label="Measured (receiver)", rasterized=True)
    ax.axhline(y=np.median(recv_bw), color="red", linestyle="--", linewidth=0.8,
               label=f"Median: {np.median(recv_bw):.1f} Gbps")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0, top=max(np.max(recv_bw), link_capacity_gbps) * 1.08)
    ax.set_title("Measured BW (receiver, per-window)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # Row 2: First 10 min
    # mask_10m = recv_times <= 600
    mask_10m = recv_times <= 5 # temporarily 5 secs
    ax = axes[1, 0]
    if trace_name:
        # t_mask = t_times <= 600
        t_mask = t_times <= 5 # temporarily 5 secs
        ax.plot(t_times[t_mask], t_scaled[t_mask], linewidth=0.3,
                color="darkorange", alpha=0.6, label="Target", rasterized=True)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0, top=link_capacity_gbps * 1.08)
    # ax.set_title("Target: first 10 min", fontsize=11)
    ax.set_title("Target: first 5 secs", fontsize=11) # temporarily 5 secs
    if trace_name:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    ax.plot(recv_times[mask_10m], recv_bw[mask_10m], linewidth=0.3,
            color="steelblue", alpha=0.8, label="Measured", rasterized=True)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0, top=max(np.max(recv_bw), link_capacity_gbps) * 1.08)
    # ax.set_title("Measured: first 10 min", fontsize=11)
    ax.set_title("Measured: first 5 secs", fontsize=11) # temporarily 5 secs
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Trace Replay: Target vs Measured",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = csv_path.rsplit(".", 1)[0] + "_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot raw trace or receiver CSV")
    parser.add_argument("--trace", type=str, default=None,
                        help="Dataset name to plot raw trace")
    parser.add_argument("--receiver-csv", type=str, default=None,
                        help="Receiver CSV to plot measured BW")
    parser.add_argument("--link-capacity-gbps", type=float, default=200,
                        help="Link capacity for scaling (default: 200)")
    parser.add_argument("--all", action="store_true",
                        help="Plot all four traces")
    args = parser.parse_args()

    if args.all:
        for name in TRACE_DATASETS:
            plot_raw_trace(name, args.link_capacity_gbps)
    elif args.trace:
        plot_raw_trace(args.trace, args.link_capacity_gbps)

    if args.receiver_csv:
        plot_receiver_csv(args.receiver_csv, args.trace, args.link_capacity_gbps)

    if not args.trace and not args.receiver_csv and not args.all:
        parser.print_help()
