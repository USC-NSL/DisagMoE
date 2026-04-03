#!/usr/bin/env python3
"""
BW Profiler — read HCA hardware counters from sysfs to measure link
bandwidth with ZERO extra traffic on the wire.

Reads port_xmit_data / port_rcv_data from /sys/class/infiniband/<dev>/ports/<port>/counters/.
These counters report data in units of 4 bytes.

Optionally reads the remote peer's counters over a persistent SSH session.

Usage:
  python profile_bw.py --peer-host sgpu2 --peer-ip 10.0.0.2 --duration 300
  python profile_bw.py --peer-host sgpu2 --peer-ip 10.0.0.2 --duration 3600 \
      --output bw_profile.csv
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import threading

NET_DEVICE = "mlx5_1:1"
SSH_OPTS = "-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
DEFAULT_INTERVAL = 0.010  # 10 ms

_procs = []
_csv_path = None


def cleanup():
    for proc in _procs:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    for proc in _procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def signal_handler(signum, frame):
    cleanup()
    if _csv_path and os.path.exists(_csv_path):
        plot_profile(_csv_path)
    sys.exit(0)


def detect_head_ip():
    """Detect local IP on the RDMA interface (mlx5_1)."""
    for dev in ["ens1f1np1", "ibp65s0"]:
        try:
            out = subprocess.check_output(
                f"ip -4 addr show dev {dev} 2>/dev/null | grep -oP 'inet \\K[0-9.]+'",
                shell=True).decode().strip()
            if out:
                return out
        except subprocess.CalledProcessError:
            pass
    try:
        return subprocess.check_output(
            "hostname -I", shell=True).decode().strip().split()[0]
    except (subprocess.CalledProcessError, IndexError):
        return "127.0.0.1"


def parse_net_device(net_device_str):
    """Parse 'mlx5_1:1' into ('mlx5_1', '1')."""
    if ":" in net_device_str:
        dev, port = net_device_str.rsplit(":", 1)
        return dev, port
    return net_device_str, "1"


def sysfs_counter_paths(ib_dev, port):
    """Return (tx_path, rx_path) for the given IB device and port."""
    base = f"/sys/class/infiniband/{ib_dev}/ports/{port}/counters"
    return (os.path.join(base, "port_xmit_data"),
            os.path.join(base, "port_rcv_data"))


def read_local_counters(tx_path, rx_path):
    """Read TX and RX counters from sysfs. Returns (tx_bytes, rx_bytes).
    Counter values are in units of 4 bytes, so multiply by 4."""
    with open(tx_path, "r") as f:
        tx = int(f.read().strip()) * 4
    with open(rx_path, "r") as f:
        rx = int(f.read().strip()) * 4
    return tx, rx


class RemoteCounterReader:
    """Reads HCA counters from a remote host via a persistent SSH session.

    Launches a small shell loop on the remote side that reads the two
    counter files as fast as it receives newline prompts on stdin, and
    prints 'tx_val rx_val' per line.
    """

    def __init__(self, peer_host, tx_path, rx_path):
        self.peer_host = peer_host
        self.tx_path = tx_path
        self.rx_path = rx_path
        self.proc = None
        self.lock = threading.Lock()
        self._latest_tx = 0
        self._latest_rx = 0
        self._alive = False

    def start(self):
        remote_cmd = (
            f"while read line; do "
            f"tx=$(cat {self.tx_path}); rx=$(cat {self.rx_path}); "
            f"echo $tx $rx; "
            f"done"
        )
        ssh_cmd = (
            f"ssh {SSH_OPTS} {self.peer_host} "
            f"'bash -c \"{remote_cmd}\"'"
        )
        try:
            self.proc = subprocess.Popen(
                ssh_cmd, shell=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, preexec_fn=os.setsid)
            _procs.append(self.proc)
            # Test that it works with a single round-trip
            self.proc.stdin.write(b"\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if line:
                parts = line.decode().strip().split()
                if len(parts) == 2:
                    self._latest_tx = int(parts[0]) * 4
                    self._latest_rx = int(parts[1]) * 4
                    self._alive = True
                    return True
        except Exception as e:
            print(f"[profile] WARNING: remote counter reader failed: {e}",
                  file=sys.stderr)
        return False

    def read(self):
        """Request a fresh reading. Returns (tx_bytes, rx_bytes) or None."""
        if not self._alive or self.proc.poll() is not None:
            return None
        try:
            self.proc.stdin.write(b"\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                self._alive = False
                return None
            parts = line.decode().strip().split()
            if len(parts) == 2:
                tx = int(parts[0]) * 4
                rx = int(parts[1]) * 4
                self._latest_tx = tx
                self._latest_rx = rx
                return tx, rx
        except Exception:
            self._alive = False
        return None

    @property
    def alive(self):
        return self._alive


def plot_profile(csv_path):
    """Generate a 4-panel BW profile plot from the CSV with TX and RX."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[profile] matplotlib/numpy not available, skipping plot.")
        return

    times, tx_bws, rx_bws = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["wall_clock_sec"]))
            tx_bws.append(float(row["tx_gbps"]))
            rx_bws.append(float(row["rx_gbps"]))

    if len(tx_bws) < 2:
        print("[profile] Too few samples to plot.")
        return

    times = np.array(times)
    tx_gbps = np.array(tx_bws)
    rx_gbps = np.array(rx_bws)

    # Use TX for primary stats (interference sent)
    for label, bw_gbps in [("TX", tx_gbps), ("RX", rx_gbps)]:
        median_bw = np.median(bw_gbps)
        mean_bw = np.mean(bw_gbps)
        min_bw = np.min(bw_gbps)
        max_bw = np.max(bw_gbps)
        p01 = np.percentile(bw_gbps, 1)
        p99 = np.percentile(bw_gbps, 99)
        cov = np.std(bw_gbps) / mean_bw if mean_bw > 0 else 0

        worst_idx = np.argmin(bw_gbps)
        worst_time = times[worst_idx]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        color = "steelblue" if label == "TX" else "coral"

        # Panel 1: Full timeline
        ax = axes[0, 0]
        ax.plot(times, bw_gbps, linewidth=0.3, color=color, alpha=0.7,
                rasterized=True)
        ax.axhline(y=median_bw, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Median: {median_bw:.1f} Gbps")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Bandwidth (Gbps)")
        ax.set_ylim(bottom=0, top=max(max_bw * 1.05, 1))
        ax.set_title(f"{label} full profile (per-sample)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)

        # Panel 2: Zoom on worst drop (+-5s)
        ax = axes[0, 1]
        zoom_half = 5.0
        zoom_start = max(0, worst_time - zoom_half)
        zoom_end = min(times[-1], worst_time + zoom_half)
        mask = (times >= zoom_start) & (times <= zoom_end)
        if mask.sum() > 5:
            ax.plot(times[mask], bw_gbps[mask], linewidth=0.5, color=color,
                    rasterized=True)
            ax.axhline(y=median_bw, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Bandwidth (Gbps)")
            ax.set_ylim(bottom=0, top=max(max_bw * 1.05, 1))
            ax.set_title(f"Zoom: worst drop at t={worst_time:.1f}s", fontsize=10)
            ax.grid(True, alpha=0.2)
        else:
            ax.text(0.5, 0.5, "Insufficient data for zoom",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray")

        # Panel 3: BW distribution histogram
        ax = axes[1, 0]
        ax.hist(bw_gbps, bins=min(200, len(bw_gbps) // 5 + 1), color=color,
                alpha=0.7, edgecolor="none", density=True)
        ax.axvline(x=median_bw, color="red", linestyle="--", linewidth=1,
                   label=f"Median: {median_bw:.1f}")
        ax.axvline(x=p01, color="orange", linestyle=":", linewidth=1,
                   label=f"P01: {p01:.1f}")
        ax.set_xlabel("Bandwidth (Gbps)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} BW distribution", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Panel 4: Stats text
        ax = axes[1, 1]
        ax.axis("off")
        duration = times[-1] - times[0]
        stat_lines = [
            f"Direction: {label}",
            f"Samples:   {len(bw_gbps):,}",
            f"Duration:  {duration:.1f}s",
            f"Interval:  ~{duration / len(bw_gbps) * 1000:.1f} ms",
            "",
            f"Mean BW:   {mean_bw:.2f} Gbps",
            f"Median BW: {median_bw:.2f} Gbps",
            f"Std BW:    {np.std(bw_gbps):.2f} Gbps",
            f"CoV:       {cov:.6f}",
            "",
            f"Min BW:    {min_bw:.2f} Gbps",
            f"Max BW:    {max_bw:.2f} Gbps",
            f"P01:       {p01:.2f} Gbps",
            f"P99:       {p99:.2f} Gbps",
        ]
        ax.text(0.05, 0.95, "\n".join(stat_lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          alpha=0.8))
        ax.set_title(f"{label} BW statistics", fontsize=10)

        fig.suptitle(f"HCA Counter BW Profile ({label})", fontsize=13,
                     fontweight="bold", y=0.98)
        plt.tight_layout()
        suffix = f"_{label.lower()}"
        png_path = csv_path.rsplit(".", 1)[0] + f"{suffix}.png"
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[profile] Plot saved: {png_path}")

    # Also generate a combined TX+RX overlay plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(times, tx_gbps, linewidth=0.3, color="steelblue", alpha=0.7,
            label="TX", rasterized=True)
    ax.plot(times, rx_gbps, linewidth=0.3, color="coral", alpha=0.7,
            label="RX", rasterized=True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bandwidth (Gbps)")
    ax.set_ylim(bottom=0)
    ax.set_title("HCA Counter BW Profile (TX + RX)", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    png_path = csv_path.rsplit(".", 1)[0] + "_combined.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[profile] Plot saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile link BW via HCA sysfs counters (zero extra traffic).")
    parser.add_argument("--peer-host", required=True,
                        help="Peer hostname (SSH target)")
    parser.add_argument("--peer-ip", required=True,
                        help="Peer IP address (unused by counters, kept for compat)")
    parser.add_argument("--head-ip", default=None,
                        help="Local IP (auto-detected if omitted)")
    parser.add_argument("--duration", type=float, default=300,
                        help="Profile duration in seconds (default: 300)")
    parser.add_argument("--net-device", default=NET_DEVICE,
                        help=f"IB device:port (default: {NET_DEVICE})")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Sampling interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: bw_profile_<timestamp>.csv)")
    args = parser.parse_args()

    head_ip = args.head_ip or detect_head_ip()
    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"bw_profile_{ts}.csv"

    global _csv_path
    _csv_path = args.output

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ib_dev, ib_port = parse_net_device(args.net_device)
    tx_path, rx_path = sysfs_counter_paths(ib_dev, ib_port)

    # Validate that local sysfs paths exist
    for p in (tx_path, rx_path):
        if not os.path.exists(p):
            print(f"[profile] ERROR: sysfs counter not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"[profile] Head: {head_ip}, Peer: {args.peer_host} ({args.peer_ip})")
    print(f"[profile] Device: {ib_dev} port {ib_port}")
    print(f"[profile] Interval: {args.interval * 1000:.1f} ms, Duration: {args.duration}s")
    print(f"[profile] TX counter: {tx_path}")
    print(f"[profile] RX counter: {rx_path}")
    print(f"[profile] Output: {args.output}")

    # Try to set up remote counter reader
    remote = RemoteCounterReader(args.peer_host, tx_path, rx_path)
    remote_ok = remote.start()
    if remote_ok:
        print(f"[profile] Remote counter reader connected to {args.peer_host}")
    else:
        print(f"[profile] WARNING: remote counter reader unavailable, "
              f"logging local counters only")

    # Read initial counters
    prev_tx, prev_rx = read_local_counters(tx_path, rx_path)
    prev_time = time.monotonic()
    start_wall = time.time()
    start_mono = prev_time

    sample_count = 0

    print(f"[profile] Sampling started...")

    with open(args.output, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        if remote_ok:
            writer.writerow(["wall_clock_sec", "tx_gbps", "rx_gbps",
                             "remote_tx_gbps", "remote_rx_gbps"])
        else:
            writer.writerow(["wall_clock_sec", "tx_gbps", "rx_gbps"])

        # For remote, track previous values
        if remote_ok:
            prev_remote_tx = remote._latest_tx
            prev_remote_rx = remote._latest_rx

        while True:
            time.sleep(args.interval)

            now_mono = time.monotonic()
            elapsed_total = now_mono - start_mono
            if elapsed_total >= args.duration:
                break

            # Read local counters
            cur_tx, cur_rx = read_local_counters(tx_path, rx_path)
            dt = now_mono - prev_time

            if dt <= 0:
                continue

            # Compute BW in Gbps (bytes -> bits -> Gbps)
            tx_bw = (cur_tx - prev_tx) * 8.0 / dt / 1e9
            rx_bw = (cur_rx - prev_rx) * 8.0 / dt / 1e9

            # Handle counter wraps (unlikely but defensive)
            if tx_bw < 0:
                tx_bw = 0.0
            if rx_bw < 0:
                rx_bw = 0.0

            wall_sec = time.time() - start_wall

            row = [f"{wall_sec:.4f}", f"{tx_bw:.4f}", f"{rx_bw:.4f}"]

            # Read remote counters
            if remote_ok:
                remote_result = remote.read()
                if remote_result is not None:
                    r_tx, r_rx = remote_result
                    r_tx_bw = (r_tx - prev_remote_tx) * 8.0 / dt / 1e9
                    r_rx_bw = (r_rx - prev_remote_rx) * 8.0 / dt / 1e9
                    if r_tx_bw < 0:
                        r_tx_bw = 0.0
                    if r_rx_bw < 0:
                        r_rx_bw = 0.0
                    row.extend([f"{r_tx_bw:.4f}", f"{r_rx_bw:.4f}"])
                    prev_remote_tx = r_tx
                    prev_remote_rx = r_rx
                else:
                    row.extend(["", ""])

            writer.writerow(row)
            csvf.flush()

            sample_count += 1
            prev_tx = cur_tx
            prev_rx = cur_rx
            prev_time = now_mono

            if sample_count % 100 == 1:
                print(f"[profile] t={wall_sec:.1f}s  TX={tx_bw:.2f} Gbps  "
                      f"RX={rx_bw:.2f} Gbps")

    cleanup()
    print(f"\n[profile] Done. {sample_count} samples in {args.output}")

    # Print summary
    if sample_count > 0:
        tx_bws_all, rx_bws_all = [], []
        with open(args.output) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx_bws_all.append(float(row["tx_gbps"]))
                rx_bws_all.append(float(row["rx_gbps"]))

        for label, bws in [("TX", tx_bws_all), ("RX", rx_bws_all)]:
            bws.sort()
            n = len(bws)
            if n == 0:
                continue
            mean_bw = sum(bws) / n
            median_bw = bws[n // 2]
            min_bw = bws[0]
            max_bw = bws[-1]
            p01 = bws[int(n * 0.01)]
            p99 = bws[min(int(n * 0.99), n - 1)]
            print(f"[profile] {label}: mean={mean_bw:.1f} median={median_bw:.1f} "
                  f"min={min_bw:.1f} max={max_bw:.1f} "
                  f"P01={p01:.1f} P99={p99:.1f} Gbps")

        plot_profile(args.output)


if __name__ == "__main__":
    main()
