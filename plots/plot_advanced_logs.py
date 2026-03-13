#!/usr/bin/env python3
"""Plot advanced logging results: per-rank MoE batch-size CDF, execution-time CDF, queuing-delay heatmap."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import glob
import argparse
import re

# 8-color palette — visually distinct, works on white background
RANK_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


def load_moe_steps_per_device(log_dir):
    """Load moe_steps.json per device. Returns dict: device_id -> (batch_sizes, exec_times)."""
    per_device = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "device_*", "moe_steps*.json"))):
        # extract device id from path
        m = re.search(r"device_(\d+)", path)
        if not m:
            continue
        dev_id = int(m.group(1))
        with open(path) as f:
            data = json.load(f)
        bsz = np.array(data.get("batch_sizes", []))
        etime = np.array(data.get("execution_times_ms", []))
        per_device[dev_id] = (bsz, etime)
    return per_device


def load_queuing_delays(log_dir):
    """Load and aggregate queuing_delays.json from all device_* subdirs.
    Returns dict: (layer_id, expert_id) -> mean_delay_ms (averaged across devices).
    """
    agg = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "device_*", "queuing_delays*.json"))):
        with open(path) as f:
            data = json.load(f)
        for key, entry in data.items():
            lid = entry["layer_id"]
            eid = entry["expert_id"]
            k = (lid, eid)
            if k not in agg:
                agg[k] = []
            agg[k].append(entry["mean_ms"])
    result = {}
    for k, means in agg.items():
        result[k] = np.mean(means)
    return result


def plot_per_rank_cdf(ax, per_device, key_idx, title, xlabel):
    """Plot one CDF curve per device on the same axes.

    key_idx: 0 for batch_sizes, 1 for exec_times (index into the per-device tuple).
    """
    if not per_device:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    for dev_id in sorted(per_device.keys()):
        values = per_device[dev_id][key_idx]
        if len(values) == 0:
            continue
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        color = RANK_COLORS[dev_id % len(RANK_COLORS)]
        ax.plot(sorted_vals, cdf, color=color, linewidth=1.2, label=f"GPU {dev_id}")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.grid(axis="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9)


def plot_heatmap(ax, delays, title):
    if not delays:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    layers = sorted(set(k[0] for k in delays))
    experts = sorted(set(k[1] for k in delays))

    layer_idx = {l: i for i, l in enumerate(layers)}
    expert_idx = {e: i for i, e in enumerate(experts)}

    grid = np.full((len(experts), len(layers)), np.nan)
    for (lid, eid), val in delays.items():
        grid[expert_idx[eid], layer_idx[lid]] = val

    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Expert", fontsize=10)

    if len(layers) <= 40:
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=5, rotation=90)
    else:
        step = max(1, len(layers) // 20)
        ax.set_xticks(range(0, len(layers), step))
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), step)], fontsize=5, rotation=90)

    if len(experts) <= 20:
        ax.set_yticks(range(len(experts)))
        ax.set_yticklabels(experts, fontsize=6)
    else:
        step = max(1, len(experts) // 16)
        ax.set_yticks(range(0, len(experts), step))
        ax.set_yticklabels([experts[i] for i in range(0, len(experts), step)], fontsize=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Queuing Delay (ms)", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Plot advanced logging results")
    parser.add_argument("log_dir", type=str, help="Path to advanced_logs directory")
    parser.add_argument("--label", type=str, default="", help="Label suffix for output filename")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: same as log_dir)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.log_dir
    os.makedirs(out_dir, exist_ok=True)

    per_device = load_moe_steps_per_device(args.log_dir)
    delays = load_queuing_delays(args.log_dir)

    suffix = f"_{args.label}" if args.label else ""

    # --- Per-rank CDF plots (1x2) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_per_rank_cdf(axes[0], per_device, 0, "MoE Batch Size CDF (per GPU)", "Batch Size (tokens)")
    plot_per_rank_cdf(axes[1], per_device, 1, "MoE Step Execution Time CDF (per GPU)", "Time (ms)")
    fig.suptitle(f"MoE Step Diagnostics{' — ' + args.label if args.label else ''}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    cdf_path = os.path.join(out_dir, f"moe_cdf{suffix}.png")
    fig.savefig(cdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cdf_path}")

    # --- Heatmap ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    plot_heatmap(ax, delays, f"Queuing Delay Heatmap{' — ' + args.label if args.label else ''}")
    fig.tight_layout()
    heatmap_path = os.path.join(out_dir, f"queuing_heatmap{suffix}.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {heatmap_path}")


if __name__ == "__main__":
    main()
