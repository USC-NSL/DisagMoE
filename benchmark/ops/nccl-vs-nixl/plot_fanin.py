#!/usr/bin/env python3
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt


LINK_RATE_MBPS = 200e9 / 8 / 1e6


def load_results(results_dir):
    senders = {}
    receivers = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_rank*.json"))):
        with open(path) as f:
            r = json.load(f)
        key = (r["backend"], r["msg_bytes"])
        if r["role"] == "sender":
            senders.setdefault(key, []).append(r)
        elif r["role"] == "receiver":
            receivers[key] = r
    return senders, receivers


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
    senders, receivers = load_results(args.results_dir)

    if not senders:
        print("No sender result files found.")
        return

    agg = {}
    for key, runs in senders.items():
        n = len(runs)
        avg_lat = sum(r["avg_us"] for r in runs) / n
        agg_tput = sum(r["throughput_mbps"] for r in runs)
        agg_rate = sum(r["msg_rate"] for r in runs)
        rx = receivers.get(key)
        agg[key] = {
            "avg_us": avg_lat,
            "agg_throughput_mbps": agg_tput,
            "agg_msg_rate": agg_rate,
            "num_senders": n,
            "rx_throughput_mbps_hw": rx["rx_throughput_mbps_hw"] if rx else None,
            "rx_throughput_mbps_expected": rx["rx_throughput_mbps_expected"] if rx else None,
        }

    msg_sizes = sorted({k[1] for k in agg})
    backends = sorted({k[0] for k in agg})
    colors = {"nccl": "#2196F3", "nccl_gather": "#4CAF50", "uccl": "#9C27B0", "nixl": "#FF5722"}
    markers = {"nccl": "s", "nccl_gather": "D", "uccl": "o", "nixl": "^"}
    labels = {"nccl": "NCCL P2P", "nccl_gather": "NCCL Gather", "uccl": "UCCL P2P", "nixl": "NIXL"}
    xlabels = [fmt_size(s) for s in msg_sizes]

    n_senders = agg[next(iter(agg))]["num_senders"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"NCCL vs NIXL  |  Fan-In {n_senders}->1 (RoCE mlx5_1)  "
        f"|  sender self-measured vs receiver HW-counter observed",
        fontsize=13,
    )

    ax = axes[0]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in agg]
        ys = [agg[(backend, sz)]["avg_us"] for sz in xs]
        ax.plot(
            xs, ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=labels.get(backend, backend),
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
            xs, ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=labels.get(backend, backend),
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
        ys_send = [agg[(backend, sz)]["agg_throughput_mbps"] for sz in xs]
        ax.plot(
            xs, ys_send,
            f"{markers[backend]}-",
            color=colors[backend],
            label=f"{labels.get(backend, backend)} sender aggregate",
            markersize=8,
        )
        ys_rx_hw = [agg[(backend, sz)]["rx_throughput_mbps_hw"] for sz in xs]
        if all(y is not None for y in ys_rx_hw):
            ax.plot(
                xs, ys_rx_hw,
                f"{markers[backend]}--",
                color=colors[backend],
                label=f"{labels.get(backend, backend)} receiver HW-counter",
                markersize=8,
                markerfacecolor="white",
            )
    ax.axhline(
        LINK_RATE_MBPS,
        color="gray", linestyle=":", linewidth=1.5,
        label=f"200Gbps link rate ({LINK_RATE_MBPS:.0f} MB/s)",
    )
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Aggregate Throughput (MB/s)")
    ax.set_title(f"Aggregate Throughput ({n_senders} senders)")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.out_dir, "nccl_vs_nixl_fanin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")

    print()
    print("Cross-verification summary (sender-aggregate vs receiver-HW-counter):")
    print(f"{'backend':<8} {'msg_bytes':>10} {'send_agg_MBps':>14} {'recv_hw_MBps':>14} {'ratio':>6} {'%link':>6}")
    for backend in backends:
        for sz in msg_sizes:
            key = (backend, sz)
            if key not in agg:
                continue
            a = agg[key]
            send = a["agg_throughput_mbps"]
            recv = a["rx_throughput_mbps_hw"]
            if recv is not None:
                ratio = send / recv if recv else float("nan")
                pct_link = 100 * max(send, recv) / LINK_RATE_MBPS
                print(
                    f"{backend:<8} {sz:>10} {send:>14.1f} {recv:>14.1f} "
                    f"{ratio:>6.2f} {pct_link:>5.1f}%"
                )


if __name__ == "__main__":
    main()
