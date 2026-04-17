#!/usr/bin/env python3
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    data = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_sender.json"))):
        with open(path) as f:
            r = json.load(f)
        key = (r["backend"], r["msg_bytes"])
        data[key] = r
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_results(args.results_dir)

    if not data:
        print("No result files found.")
        return

    msg_sizes = sorted({k[1] for k in data})
    backends = sorted({k[0] for k in data})
    colors = {"nccl": "#2196F3", "nixl": "#FF5722"}
    markers = {"nccl": "s", "nixl": "^"}

    pipeline_depth = data[next(iter(data))].get("pipeline_depth", "?")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"NCCL vs NIXL  |  Open-Loop P2P (RoCE mlx5_1)  |  pipeline_depth={pipeline_depth}",
        fontsize=13,
    )

    ax = axes[0]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in data]
        ys = [data[(backend, sz)]["avg_us"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )

    def fmt_size(b):
        if b >= 1024 * 1024:
            return f"{b / 1024 / 1024:.0f}MB"
        if b >= 1024:
            return f"{b / 1024:.0f}KB"
        return f"{b}B"

    xlabels = [fmt_size(s) for s in msg_sizes]

    ax.set_xlabel("Message Size")
    ax.set_ylabel("Latency (us)")
    ax.set_title("Avg Amortized Latency per Message")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in data]
        ys = [data[(backend, sz)]["msg_rate"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Message Rate (msg/s)")
    ax.set_title("Sustained Message Rate")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in data]
        ys = [data[(backend, sz)]["throughput_mbps"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Throughput")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in data]
        ys = [data[(backend, sz)]["msg_rate"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("Message Rate (msg/s)")
    ax.set_title("Sustained Message Rate")
    ax.set_xticks(msg_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in data]
        ys = [data[(backend, sz)]["throughput_mbps"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Throughput")
    ax.set_xticks(msg_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.out_dir, "nccl_vs_nixl.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    ax2.set_title("Latency Distribution")
    bp_data = []
    bp_labels = []
    bp_colors = []
    for sz in msg_sizes:
        for backend in backends:
            lats = data.get((backend, sz), {}).get("latencies_us", [])
            if lats:
                bp_data.append(lats)
                bp_labels.append(f"{backend.upper()}\n{fmt_size(sz)}")
                bp_colors.append(colors[backend])
    if bp_data:
        bp = ax2.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], bp_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax2.set_ylabel("Latency (us)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path2 = os.path.join(args.out_dir, "nccl_vs_nixl_boxplot.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Box plot saved: {path2}")


if __name__ == "__main__":
    main()
