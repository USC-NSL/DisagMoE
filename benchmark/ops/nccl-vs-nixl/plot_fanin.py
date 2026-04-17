#!/usr/bin/env python3
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    data = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_rank[123]*.json"))):
        with open(path) as f:
            r = json.load(f)
        key = (r["backend"], r["msg_bytes"])
        data.setdefault(key, []).append(r)
    return data


def fmt_size(b):
    if b >= 1024 * 1024:
        return f"{b / 1024 / 1024:.0f}MB"
    if b >= 1024:
        return f"{b / 1024:.0f}KB"
    return f"{b}B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw = load_results(args.results_dir)

    if not raw:
        print("No result files found.")
        return

    agg = {}
    for key, runs in raw.items():
        n = len(runs)
        avg_lat = sum(r["avg_us"] for r in runs) / n
        agg_tput = sum(r["throughput_mbps"] for r in runs)
        agg_rate = sum(r["msg_rate"] for r in runs)
        agg[key] = {
            "avg_us": avg_lat,
            "agg_throughput_mbps": agg_tput,
            "agg_msg_rate": agg_rate,
            "num_senders": n,
        }

    msg_sizes = sorted({k[1] for k in agg})
    backends = sorted({k[0] for k in agg})
    colors = {"nccl": "#2196F3", "nixl": "#FF5722"}
    markers = {"nccl": "s", "nixl": "^"}
    xlabels = [fmt_size(s) for s in msg_sizes]

    n_senders = agg[next(iter(agg))]["num_senders"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"NCCL vs NIXL  |  Fan-In {n_senders}→1 (RoCE mlx5_1)  |  per-send latency",
        fontsize=13,
    )

    ax = axes[0]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in agg]
        ys = [agg[(backend, sz)]["avg_us"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Latency (us)")
    ax.set_title("Avg Per-Sender Latency")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in agg]
        ys = [agg[(backend, sz)]["agg_msg_rate"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Aggregate Message Rate (msg/s)")
    ax.set_title(f"Aggregate Message Rate ({n_senders} senders)")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in agg]
        ys = [agg[(backend, sz)]["agg_throughput_mbps"] for sz in xs]
        ax.plot(
            xs,
            ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Aggregate Throughput (MB/s)")
    ax.set_title(f"Aggregate Throughput ({n_senders} senders)")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.out_dir, "nccl_vs_nixl_fanin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")


if __name__ == "__main__":
    main()
