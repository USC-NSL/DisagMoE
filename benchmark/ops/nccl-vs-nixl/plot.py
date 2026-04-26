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
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        key = (r["backend"], r["msg_bytes"])
        if r["role"] == "sender":
            senders[key] = r
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

    msg_sizes = sorted({k[1] for k in senders})
    backends = sorted({k[0] for k in senders})
    colors = {"nccl": "#2196F3", "nixl": "#FF5722"}
    markers = {"nccl": "s", "nixl": "^"}
    xlabels = [fmt_size(s) for s in msg_sizes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "NCCL vs NIXL  |  1:1 P2P (RoCE mlx5_1)  "
        "|  sender self-measured vs receiver HW-counter observed",
        fontsize=13,
    )

    ax = axes[0]
    for backend in backends:
        xs = [sz for sz in msg_sizes if (backend, sz) in senders]
        ys = [senders[(backend, sz)]["avg_us"] for sz in xs]
        ax.plot(
            xs, ys,
            f"{markers[backend]}-",
            color=colors[backend],
            label=backend.upper(),
            markersize=8,
        )
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
        xs = [sz for sz in msg_sizes if (backend, sz) in senders]
        ys = [senders[(backend, sz)]["msg_rate"] for sz in xs]
        ax.plot(
            xs, ys,
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
        xs = [sz for sz in msg_sizes if (backend, sz) in senders]
        ys_send = [senders[(backend, sz)]["throughput_mbps"] for sz in xs]
        ax.plot(
            xs, ys_send,
            f"{markers[backend]}-",
            color=colors[backend],
            label=f"{backend.upper()} sender",
            markersize=8,
        )
        ys_rx_hw = [
            receivers.get((backend, sz), {}).get("rx_throughput_mbps_hw") for sz in xs
        ]
        if all(y is not None for y in ys_rx_hw):
            ax.plot(
                xs, ys_rx_hw,
                f"{markers[backend]}--",
                color=colors[backend],
                label=f"{backend.upper()} receiver HW-counter",
                markersize=8,
                markerfacecolor="white",
            )
    ax.axhline(
        LINK_RATE_MBPS,
        color="gray", linestyle=":", linewidth=1.5,
        label=f"200Gbps link rate ({LINK_RATE_MBPS:.0f} MB/s)",
    )
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Throughput")
    ax.set_xticks(msg_sizes)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.out_dir, "nccl_vs_nixl.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")

    print()
    print("Cross-verification summary (sender self vs receiver HW-counter):")
    print(f"{'backend':<8} {'msg_bytes':>10} {'send_MBps':>11} {'recv_hw_MBps':>14} {'ratio':>6} {'%link':>6}")
    for backend in backends:
        for sz in msg_sizes:
            if (backend, sz) not in senders:
                continue
            send = senders[(backend, sz)]["throughput_mbps"]
            recv_obj = receivers.get((backend, sz))
            if recv_obj is None:
                continue
            recv = recv_obj["rx_throughput_mbps_hw"]
            ratio = send / recv if recv else float("nan")
            pct_link = 100 * max(send, recv) / LINK_RATE_MBPS
            print(
                f"{backend:<8} {sz:>10} {send:>11.1f} {recv:>14.1f} "
                f"{ratio:>6.2f} {pct_link:>5.1f}%"
            )


if __name__ == "__main__":
    main()
