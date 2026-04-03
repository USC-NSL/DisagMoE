#!/usr/bin/env python3
"""
Trace-Driven Workload Traffic Simulator — replay De Sensi et al. network
noise traces as real UCX (TCP) traffic with sub-millisecond fidelity.

Two modes:
  trace    — Replay a real noise trace by computing per-window target BW
             rates.  The C sender (ucx_sender) paces sends with a token
             bucket at the exact rate per window.
  constant — Steady traffic at a fixed rate (bytes/sec or Gbps).

Architecture:
  --role server   Launches ucx_receiver (C binary) as the listener.
                  Run on the REMOTE/PEER node.
  --role client   Generates a binary rate schedule (V2), then launches
                  ucx_sender (C binary) which reads the schedule and
                  shapes traffic with a token bucket per window.
                  Run LOCALLY on the head node.

Schedule file format (V2):
  Header (32 bytes): magic "SCHD", version=2, num_windows, msg_size,
                     window_ns, pad
  Body:              num_windows × double (rate in bytes/sec)

The token bucket gives continuous, smooth rate shaping — NOT on/off
duty-cycling.  The instantaneous BW matches the target at every point.

Trace scaling:
  The trace measures workload BW (high = healthy, low = noisy).
  We generate interference as the complement:
    - workload_bw = (sample_bw / max_bw) × link_capacity
    - interference_rate = link_capacity − workload_bw
  So high trace BW → low interference, low trace BW → high interference.

Usage:
  # Trace replay — Oracle HPC noise pattern on our 200G link:
  python interfere.py --role server --peer-ip 10.0.0.1
  python interfere.py --role client --peer-ip 10.0.0.1 --mode trace \\
      --trace oracle_hpc --link-capacity-gbps 200

  # Constant rate — steady 10 Gbps:
  python interfere.py --role server --peer-ip 10.0.0.1 --duration 300
  python interfere.py --role client --peer-ip 10.0.0.1 --mode constant \\
      --target-rate-gbps 10 --duration 300

Available trace datasets:
  oracle_hpc      Oracle HPC RDMA (rare severe clustered bursts)
  aws_hpc_metal   AWS HPC Metal EFA (persistent moderate jitter)
  azure_hpc_200g  Azure HPC 200G HDR IB (rare mild dips)
  deep_est_ib     DEEP-EST IB EDR (near-zero, control baseline)
"""

import argparse
import csv
import os
import signal
import struct
import subprocess
import sys
from statistics import median


# ---------------------------------------------------------------------------
# Trace dataset name → raw trace file path (relative to script directory)
# ---------------------------------------------------------------------------

TRACE_DATASETS = {
    "oracle_hpc": "raw_traces/2022_07_13_13_09_16/ng_netnoise_mpi_bw.out",
    "aws_hpc_metal": "raw_traces/2022_05_15_18_54_34/ng_netnoise_mpi_bw.out",
    "azure_hpc_200g": "raw_traces/2022_03_25_17_01_24/ng_netnoise_mpi_bw.out",
    "deep_est_ib": "raw_traces/2022_05_19_16_35_07/ng_netnoise_mpi_bw.out",
}

TRACE_DURATION_SEC = 3600   # Each trace is a 1-hour netgauge run
TRACE_MSG_SIZE = 16777216   # 16 MB messages in De Sensi et al. traces
TRACE_WARMUP = 20           # Skip first 20 of every 1000 samples


def resolve_trace_path(trace_arg):
    """Resolve dataset name or file path to actual trace file path."""
    if trace_arg in TRACE_DATASETS:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, TRACE_DATASETS[trace_arg])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Trace file not found: {path}\n"
                f"Run: git clone https://github.com/DanieleDeSensi/cloud_noise_data.git "
                f"and copy traces to raw_traces/")
        return path
    elif os.path.isfile(trace_arg):
        return trace_arg
    else:
        names = ", ".join(TRACE_DATASETS.keys())
        raise FileNotFoundError(
            f"'{trace_arg}' is not a known dataset ({names}) "
            f"and is not a valid file path")


def load_trace_rates(trace_path, link_capacity_bps, window_ms=1):
    """Load raw trace, compute per-window target rates in bytes/sec.

    Returns (rates_bps, window_sec).

    Scaling strategy:
      1. Convert each RTT sample to workload BW (bytes/sec).
      2. Use max of the trace BW distribution as the "no noise"
         reference point (max → link_capacity).
      3. Interference = complement: rate = (1 − sample_bw/max_bw) × link_capacity.
         High workload BW → low interference; low workload BW → high interference.
    """
    # Parse RTTs from raw trace
    rtts = []
    with open(trace_path) as f:
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

    # Exclude warmup: first 20 of every 1000 samples
    cleaned = [rtt for i, rtt in enumerate(rtts) if (i % 1000) >= TRACE_WARMUP]
    if not cleaned:
        raise ValueError(f"No valid samples in trace: {trace_path}")

    # Compute BW in bytes/sec for each sample
    # RTT is in microseconds, MSG_SIZE in bytes
    bw_bps = [TRACE_MSG_SIZE / (rtt * 1e-6) for rtt in cleaned]

    # Reference statistics
    med_bw = median(bw_bps)
    max_bw = max(bw_bps)

    # Compute window parameters
    sample_interval = TRACE_DURATION_SEC / len(bw_bps)
    samples_per_window = max(1, round(window_ms / 1000.0 / sample_interval))
    window_sec = samples_per_window * sample_interval

    # Map each window's BW to a scaled rate using max as reference
    rates = []
    for w in range(0, len(bw_bps), samples_per_window):
        chunk = bw_bps[w:w + samples_per_window]
        mean_bw = sum(chunk) / len(chunk)

        # Trace BW = workload's observed BW.  Interference = complement.
        # rate = link_capacity − (trace_bw / trace_max) × link_capacity
        ratio = mean_bw / max_bw
        rate = (1.0 - ratio) * link_capacity_bps
        rates.append(rate)

    # Stats
    avg_rate = sum(rates) / len(rates) if rates else 0
    min_rate = min(rates) if rates else 0
    max_rate = max(rates) if rates else 0
    below_median = sum(1 for r in rates if r < avg_rate)

    print(f"[interfere] Trace loaded: {len(bw_bps):,} samples, "
          f"median={med_bw/1e9:.2f} GB/s, max={max_bw/1e9:.2f} GB/s")
    print(f"[interfere] Scaling: max → {link_capacity_bps*8/1e9:.0f} Gbps "
          f"(link capacity)")
    print(f"[interfere] Schedule: {len(rates)} windows × "
          f"{window_sec*1000:.2f}ms, "
          f"avg={avg_rate*8/1e9:.1f} Gbps, "
          f"min={min_rate*8/1e9:.1f}, max={max_rate*8/1e9:.1f} Gbps")

    return rates, window_sec


# ---------------------------------------------------------------------------
# Binary schedule file I/O (V2: rate-based)
# ---------------------------------------------------------------------------

SCHED_MAGIC = b"SCHD"
SCHED_VERSION_V2 = 2
SCHED_HEADER_FMT = "<4sIIIQ"  # magic(4s) version(I) num_windows(I)
                               # msg_size(I) window_ns(Q)
SCHED_HEADER_SIZE = 32         # 24 bytes struct + 8 bytes pad


def write_schedule_v2(rates_bps, window_sec, msg_size, path):
    """Write V2 binary schedule file (rate-based) for ucx_sender."""
    num_windows = len(rates_bps)
    window_ns = int(window_sec * 1e9)

    header = struct.pack(SCHED_HEADER_FMT,
                         SCHED_MAGIC,
                         SCHED_VERSION_V2,
                         num_windows,
                         msg_size,
                         window_ns)
    header += b'\x00' * 8  # pad to 32 bytes

    # V2: double per window (8 bytes each)
    data = struct.pack(f'<{num_windows}d', *rates_bps)

    with open(path, 'wb') as f:
        f.write(header)
        f.write(data)

    avg = sum(rates_bps) / num_windows if num_windows else 0
    total_sec = num_windows * window_sec
    file_size = SCHED_HEADER_SIZE + num_windows * 8

    print(f"[interfere] V2 schedule written: {num_windows} windows × "
          f"{window_sec*1000:.2f}ms = {total_sec:.0f}s, "
          f"avg_rate={avg*8/1e9:.1f} Gbps, "
          f"file={file_size/1024:.0f} KB → {path}")

    return path


# ---------------------------------------------------------------------------
# Calibration curve: map desired interference → TCP payload rate
# ---------------------------------------------------------------------------

def load_calibration_curve(csv_path):
    """Load calibration CSV and return sorted list of (actual_interference_bps,
    tcp_payload_bps) pairs for interpolation.

    CSV columns: tcp_payload_gbps, actual_interference_gbps
    We include an implicit (0, 0) origin point.
    """
    points = [(0.0, 0.0)]  # origin: zero payload → zero interference
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tcp_gbps = float(row["tcp_payload_gbps"])
            interf_gbps = float(row["actual_interference_gbps"])
            # Convert to bytes/sec
            tcp_bps = tcp_gbps * 1e9 / 8.0
            interf_bps = interf_gbps * 1e9 / 8.0
            points.append((interf_bps, tcp_bps))

    # Sort by actual interference (x-axis for interpolation)
    points.sort(key=lambda p: p[0])
    return points


def calibrated_rate(desired_interference_bps, cal_curve):
    """Given a desired interference rate (bytes/sec), return the TCP payload
    rate (bytes/sec) needed to achieve it, using linear interpolation on the
    calibration curve.

    cal_curve: sorted list of (actual_interference_bps, tcp_payload_bps).
    """
    if desired_interference_bps <= 0:
        return 0.0

    # Find bracketing points
    for i in range(len(cal_curve) - 1):
        i0, p0 = cal_curve[i]
        i1, p1 = cal_curve[i + 1]
        if i0 <= desired_interference_bps <= i1:
            if i1 == i0:
                return p0
            t = (desired_interference_bps - i0) / (i1 - i0)
            return p0 + t * (p1 - p0)

    # Extrapolate beyond last calibration point (linear from last two points)
    if len(cal_curve) >= 2:
        i0, p0 = cal_curve[-2]
        i1, p1 = cal_curve[-1]
        if i1 != i0:
            slope = (p1 - p0) / (i1 - i0)
            return p1 + slope * (desired_interference_bps - i1)
    # Fallback: return as-is
    return desired_interference_bps


# ---------------------------------------------------------------------------
# Binary paths
# ---------------------------------------------------------------------------

def _find_binary(name):
    """Find ucx_sender or ucx_receiver binary."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, name)
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    try:
        result = subprocess.check_output(["which", name],
                                         stderr=subprocess.DEVNULL)
        return result.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# Server role: launch ucx_receiver
# ---------------------------------------------------------------------------

def _ucx_env(transport="rc", net_device="mlx5_1:1"):
    env = os.environ.copy()
    env["UCX_TLS"] = transport
    if transport == "tcp":
        env["UCX_TCP_SNDBUF"] = "4194304"
        env["UCX_TCP_RCVBUF"] = "4194304"
        env["UCX_TCP_NODELAY"] = "y"
    else:
        env["UCX_NET_DEVICES"] = net_device
    return env


def _run_server(args):
    """Launch N ucx_receiver instances on consecutive ports."""
    receiver = _find_binary("ucx_receiver")
    if not receiver:
        print("[interfere] ERROR: ucx_receiver binary not found.\n"
              "  Run: ./build_ucx.sh", file=sys.stderr)
        sys.exit(1)

    n = args.num_streams
    env = _ucx_env(args.ucx_transport, args.net_device)
    procs = []

    print(f"[interfere] Launching {n} ucx_receiver streams "
          f"on ports {args.base_port}–{args.base_port + n - 1}")

    for i in range(n):
        port = args.base_port + i
        cmd = [receiver, str(port)]
        if args.duration > 0:
            cmd += ["--duration", str(int(args.duration + 120))]
        cmd += ["--max-msg", str(args.msg_size)]
        if hasattr(args, 'csv') and args.csv and i == 0:
            cmd += ["--csv", args.csv]
        procs.append(subprocess.Popen(cmd, env=env))

    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n[interfere] Interrupted, stopping receivers...")
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    print("[interfere] Receivers done.")


# ---------------------------------------------------------------------------
# Client role: generate schedule + launch ucx_sender
# ---------------------------------------------------------------------------

def _run_client(args):
    """Generate rate schedule and launch N ucx_sender instances."""
    sender = _find_binary("ucx_sender")
    if not sender:
        print("[interfere] ERROR: ucx_sender binary not found.\n"
              "  Run: ./build_ucx.sh", file=sys.stderr)
        sys.exit(1)

    n = args.num_streams

    # Compute per-window rates (bytes/sec)
    if args.mode == "trace":
        rates, window_sec = _compute_trace_rates(args)
    else:
        rates, window_sec = _compute_constant_rates(args)

    if not rates:
        print("[interfere] Empty schedule, nothing to do.")
        return

    if args.congestion_multiplier != 1.0:
        cap = args.link_capacity * 1e9 / 8 if args.link_capacity else float('inf')
        rates = [min(r * args.congestion_multiplier, cap) for r in rates]
        print(f"[interfere] Applied congestion multiplier: {args.congestion_multiplier}x")

    # Apply calibration: per-rate interpolation from sweep, or uniform scaling
    if args.calibration_csv and os.path.isfile(args.calibration_csv):
        cal_curve = load_calibration_curve(args.calibration_csv)
        print(f"[interfere] Applying calibration curve from "
              f"{args.calibration_csv} ({len(cal_curve)-1} points)")
        rates = [calibrated_rate(r, cal_curve) for r in rates]
    elif args.tcp_overhead > 1.0:
        print(f"[interfere] Applying TCP overhead correction: "
              f"{args.tcp_overhead:.4f}x")
        rates = [r / args.tcp_overhead for r in rates]

    # Divide rates across N streams
    per_stream_rates = [r / n for r in rates]

    # Write V2 schedule (shared by all streams)
    schedule_path = f"/tmp/ucx_schedule_{os.getpid()}.bin"
    write_schedule_v2(per_stream_rates, window_sec, args.msg_size,
                      schedule_path)

    env = _ucx_env(args.ucx_transport, args.net_device)
    procs = []
    print(f"[interfere] Launching {n} ucx_sender streams "
          f"to {args.peer_ip} ports {args.base_port}–{args.base_port + n - 1}")

    for i in range(n):
        port = args.base_port + i
        cmd = [sender, args.peer_ip, str(port),
               "--schedule", schedule_path]
        if args.duration > 0:
            cmd += ["--duration", str(int(args.duration))]
        procs.append(subprocess.Popen(cmd, env=env))

    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n[interfere] Interrupted, stopping senders...")
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        try:
            os.unlink(schedule_path)
        except OSError:
            pass

    print("[interfere] Senders done.")


# ---------------------------------------------------------------------------
# Schedule computation: trace mode
# ---------------------------------------------------------------------------

def _compute_trace_rates(args):
    """Load De Sensi trace and compute per-window target rates."""
    try:
        trace_path = resolve_trace_path(args.trace)
    except (OSError, ValueError) as e:
        print(f"[interfere] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Link capacity in bytes/sec
    link_capacity_bps = args.link_capacity_gbps * 1e9 / 8.0

    rates, window_sec = load_trace_rates(
        trace_path, link_capacity_bps,
        window_ms=args.trace_window_ms)

    return rates, window_sec


# ---------------------------------------------------------------------------
# Schedule computation: constant mode
# ---------------------------------------------------------------------------

def _compute_constant_rates(args):
    """Compute constant rate schedule."""
    window_sec = 0.001  # 1ms

    if args.target_rate_gbps is not None:
        rate_bps = args.target_rate_gbps * 1e9 / 8.0  # Gbps → bytes/sec
        print(f"[interfere] Constant mode: {args.target_rate_gbps:.1f} Gbps "
              f"= {rate_bps/1e9:.2f} GB/s")
    elif args.link_capacity_gbps is not None:
        rate_bps = args.link_capacity_gbps * 1e9 / 8.0
        print(f"[interfere] Constant mode: full link capacity "
              f"({args.link_capacity_gbps:.0f} Gbps)")
    else:
        print("[interfere] WARNING: no rate specified, using 25 GB/s default",
              file=sys.stderr)
        rate_bps = 25e9

    duration = args.duration if args.duration > 0 else 3600
    num_windows = int(duration / window_sec) + 1
    rates = [rate_bps] * num_windows

    return rates, window_sec


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_proc = None


def _signal_handler(signum, frame):
    print(f"\n[interfere] Caught signal {signum}, shutting down...")
    if _proc and _proc.poll() is None:
        _proc.terminate()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _proc

    parser = argparse.ArgumentParser(
        description="Trace-driven workload traffic simulator using UCX (TCP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trace replay — Oracle HPC noise on a 200G link:
  python interfere.py --role server --peer-ip 10.0.0.1
  python interfere.py --role client --peer-ip 10.0.0.1 --mode trace \\
      --trace oracle_hpc --link-capacity-gbps 200

  # Constant rate — 10 Gbps for 5 minutes:
  python interfere.py --role server --peer-ip 10.0.0.1 --duration 300
  python interfere.py --role client --peer-ip 10.0.0.1 --mode constant \\
      --target-rate-gbps 10 --duration 300

Available dataset names for --trace:
  oracle_hpc      Oracle HPC RDMA (rare severe clustered bursts)
  aws_hpc_metal   AWS HPC Metal EFA (persistent moderate jitter)
  azure_hpc_200g  Azure HPC 200G HDR IB (rare mild dips)
  deep_est_ib     DEEP-EST IB EDR (near-zero, control baseline)
""")

    parser.add_argument("--role", required=True, choices=["server", "client"],
                        help="server=receiver (peer), client=sender (head)")
    parser.add_argument("--peer-ip", type=str, required=True,
                        help="Peer IP address")
    parser.add_argument("--mode", choices=["trace", "constant"],
                        default="trace",
                        help="Traffic mode (default: trace)")

    # Rate control
    parser.add_argument("--target-rate-gbps", type=float, default=None,
                        help="Target sending rate in Gbps (constant mode)")
    parser.add_argument("--link-capacity-gbps", type=float, default=None,
                        help="Link capacity in Gbps (e.g., 200). "
                             "Used to scale trace rates to your link.")

    # Trace mode
    parser.add_argument("--trace", type=str, default=None,
                        help="Dataset name or path to trace file")
    parser.add_argument("--trace-window-ms", type=float, default=1,
                        help="Window size in ms for trace replay (default: 1)")

    # Common
    parser.add_argument("--duration", type=float, default=0,
                        help="Run duration in seconds (0=auto: 3600 for trace)")
    parser.add_argument("--net-device", type=str, default="mlx5_1:1",
                        help="UCX_NET_DEVICES (default: mlx5_1:1)")
    parser.add_argument("--ucx-transport", type=str, default="rc",
                        help="UCX_TLS transport: rc (RDMA) or tcp (default: rc)")
    parser.add_argument("--congestion-multiplier", type=float, default=1.0,
                        help="Multiply trace interference rates by this factor (default: 1.0)")
    parser.add_argument("--base-port", type=int, default=18515,
                        help="UCX listener port (default: 18515)")
    parser.add_argument("--msg-size", type=int, default=65536,
                        help="Message size in bytes (default: 64KB)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Receiver CSV output path (server role only)")
    parser.add_argument("--tcp-overhead", type=float, default=1.0,
                        help="TCP overhead factor from calibration. "
                             "Rates are divided by this to compensate for "
                             "TCP wire overhead (default: 1.0 = no correction)")
    parser.add_argument("--calibration-csv", type=str, default=None,
                        help="Path to calibration CSV from sweep. "
                             "Overrides --tcp-overhead when provided.")
    parser.add_argument("--num-streams", type=int, default=8,
                        help="Number of parallel TCP streams for throughput. "
                             "Each stream runs a separate sender/receiver "
                             "pair on consecutive ports (default: 8)")

    args = parser.parse_args()

    # --- Auto-fallback: trace mode without --trace → constant ---
    if args.mode == "trace" and args.trace is None:
        args.mode = "constant"

    # --- Default duration ---
    if args.duration <= 0:
        if args.mode == "trace":
            args.duration = TRACE_DURATION_SEC
            print(f"[interfere] Trace mode: default duration={args.duration}s")
        else:
            print("[interfere] No --duration set, will run until Ctrl+C.",
                  file=sys.stderr)
            args.duration = float("inf")

    # --- Validation ---
    if args.mode == "trace" and args.role == "client":
        if args.link_capacity_gbps is None:
            parser.error("--link-capacity-gbps is required for trace mode")

    # --- Require the relevant UCX binary ---
    needed = "ucx_receiver" if args.role == "server" else "ucx_sender"
    if not _find_binary(needed):
        print(f"[interfere] ERROR: {needed} not found.\n"
              f"  Run: ./build_ucx.sh", file=sys.stderr)
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.role == "server":
        _run_server(args)
    else:
        _run_client(args)


if __name__ == "__main__":
    main()
