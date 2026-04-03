# Trace-Driven Network Interference Generator

Generate **interference traffic** that reproduces the BW degradation observed in real cloud/HPC noise traces. Sends rate-shaped UCX traffic at the **complement** of the trace's measured workload BW, so that a co-located workload experiences the same available-BW pattern as the original trace.

Supports **two-node** (single link) and **ring topology** (N nodes, full-duplex) modes, both using **TCP transport** with automatic calibration.

For the actual usage, please wait 2minutes after starting the script to allow the calibration sweep to complete and the interference traffic to ramp up.

## How It Works

Python (`interfere.py`) converts a De Sensi et al. noise trace into a **binary rate schedule**: one target rate (bytes/sec) per window. C programs read this schedule and pace sends with a token bucket at ns resolution.

1. **Trace mode** (default): Load a raw trace, convert per-sample RTTs to workload BW, compute the **interference rate** as `link_capacity − workload_BW`, and write per-window target rates. The sender pushes exactly the amount of traffic needed to reduce available BW to match the trace.
2. **Constant mode**: All windows get the same target rate (in Gbps).

### Trace Scaling

Each trace sample is an RTT for a 16MB message. Workload BW = 16MB / RTT.

We scale using the **max** of the trace BW distribution as the "no noise" reference point, then compute the **interference** as the complement:
```
workload_bw = (sample_bw / max_bw) × link_capacity
interference_rate = link_capacity − workload_bw
```

- When the trace shows **high workload BW** (no noise) → interference is **low** (near zero)
- When the trace shows **low workload BW** (heavy noise) → interference is **high** (fills the link)

The full dynamic range maps exactly into `[0, link_capacity]` with zero clamping.

### Token Bucket Rate Control

- **ns-resolution refill** from `CLOCK_MONOTONIC`
- Per-window rate update **without token reset** (carryover smooths window edges)
- Spin-wait with `ucp_worker_progress()` interleaving for sub-µs accuracy
- Burst limit caps token accumulation (prevents mega-bursts after idle)
- 64KB messages → ~380 msgs/ms at full speed for fine-grained pacing

## Files

- `run_interfere.sh` — **two-node runner**: calibration sweep + launches receiver + sender across two nodes via SSH (TCP)
- `run_ring.sh` — **ring topology runner**: calibration sweep + launches interference across N nodes in a ring (full-duplex, TCP)
- `interfere.py` — orchestrator: converts traces to rate schedule, applies calibration curve, launches UCX binaries
- `ucx_sender.c` — C sender: token-bucket rate-shaped sending (used by `run_interfere.sh`)
- `ucx_receiver.c` — C receiver: accepts UCX tag messages, tracks per-1ms BW (used by `run_interfere.sh`)
- `ucx_ring_node.c` — **combined sender+receiver** in a single UCX context (used by `run_ring.sh`); includes auto-reconnection on connection drops
- `build_ucx.sh` — compiles all UCX binaries
- `profile_bw.py` — **BW profiler**: reads HCA/netdev sysfs counters for zero-overhead link utilization monitoring (opt-in via `--profile`)
- `raw_traces/` — raw `ng_netnoise_mpi_bw.out` trace files (1-hour netgauge runs)

## Quick Start

### 0. Build

```bash
./build_ucx.sh
```

### 1. Two-node

Single link interference using TCP transport. A calibration sweep runs automatically at startup:

```bash
# Trace replay — AWS HPC noise for 1 hour:
./run_interfere.sh --peer-host sgpu2 --peer-ip 10.0.0.2 \
    --trace aws_hpc_metal --link-capacity-gbps 200

# Constant rate — steady 10 Gbps for 5 minutes:
./run_interfere.sh --peer-host sgpu2 --peer-ip 10.0.0.2 \
    --mode constant --target-rate-gbps 10 --duration 300

# Skip calibration and reuse a previous calibration CSV:
./run_interfere.sh --peer-host sgpu2 --peer-ip 10.0.0.2 \
    --calibration-csv /tmp/tcp_calibration.csv \
    --trace oracle_hpc --link-capacity-gbps 200

# Stop all on both nodes:
./run_interfere.sh --peer-host sgpu2 --stop
```

### 2. Ring topology (N nodes, full-duplex)

Every node both sends and receives, creating full-duplex interference on all links.
Calibration runs between nodes[0] and nodes[1] before starting:

```bash
# Ring: sgpu0→sgpu2→sgpu3→sgpu4→sgpu0 (each node sends AND receives)
./run_ring.sh --nodes sgpu0:10.0.0.1,sgpu2:10.0.0.2,sgpu3:10.0.0.3,sgpu4:10.0.0.4 \
    --trace aws_hpc_metal --link-capacity-gbps 200

# Constant 10 Gbps on every link:
./run_ring.sh --nodes sgpu0:10.0.0.1,sgpu2:10.0.0.2,sgpu3:10.0.0.3,sgpu4:10.0.0.4 \
    --mode constant --target-rate-gbps 10 --duration 300

# Stop all:
./run_ring.sh --nodes sgpu0:10.0.0.1,sgpu2:10.0.0.2,sgpu3:10.0.0.3,sgpu4:10.0.0.4 --stop
```

Both scripts automatically:
- **Calibrate TCP overhead** by sweeping 0%–150% link utilization with `ib_write_bw` using 8 parallel streams (~2 min). **The actual interference begins after calibration completes**, so expect a ~2 minute warmup before interference traffic starts.
- Generate the schedule with calibration-corrected rates
- Kill any existing interference before starting
- Wait for TCP ports to be freed from previous runs
- **Monitor actual interference** every 30s via `ib_write_bw`, reporting available BW vs baseline and implied interference
- Report per-node TX/RX Gbps periodically (ring mode)

### 3. Profile BW (standalone)

Zero-overhead link utilization monitoring via sysfs counters:

```bash
python profile_bw.py --peer-host sgpu2 --peer-ip 10.0.0.2 --duration 300
```

---

## Available Traces

| Dataset name | Noise character | Duration |
|-------------|----------------|----------|
| `oracle_hpc` | Rare severe clustered bursts (CoV=0.0076) | 1 hour |
| `aws_hpc_metal` | Persistent moderate jitter (CoV=0.039) | 1 hour |
| `azure_hpc_200g` | Rare mild dips (CoV=0.0077) | 1 hour |
| `deep_est_ib` | Near-zero control (CoV=0.0009) | 1 hour |

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `trace` | `trace` or `constant` |
| `--trace` | — | Dataset name or trace file path |
| `--link-capacity-gbps` | — | Link capacity in Gbps (scales trace rates; also triggers calibration) |
| `--target-rate-gbps` | — | Target sending rate in Gbps (constant mode) |
| `--duration` | `3600` | Experiment duration in seconds |
| `--trace-window-ms` | `1` | Window size for trace replay |
| `--net-device` | `mlx5_1:1` | IB device for calibration (e.g., `mlx5_1:1`) |
| `--base-port` | `18515` | UCX listener port |
| `--msg-size` | `65536` | Message size in bytes (64KB) |
| `--num-streams` | `8` | Parallel TCP streams per link for throughput |
| `--skip-calibration` | — | Skip the calibration sweep |
| `--tcp-overhead` | `1.0` | Uniform TCP overhead factor (skips calibration) |
| `--calibration-csv` | — | Path to a saved calibration CSV (skips calibration) |

## Architecture

Both modes use **TCP transport** (`UCX_TLS=tcp`) for stability. A calibration sweep at startup compensates for the non-linear relationship between TCP payload rate and actual link interference.

### TCP Calibration

TCP traffic causes less interference than its payload rate would suggest (wire overhead, congestion control behavior). The calibration sweep measures this empirically:

1. **Baseline**: Run `ib_write_bw` alone to measure max achievable RDMA BW → `B_baseline`
2. **Sweep 0%–150%**: For each level (0, 10, 20, ..., 90, 100, 110, 120, 130, 140, 150% of link capacity), run 8 parallel TCP streams at that rate while measuring `ib_write_bw` → `B_loaded`. Record `actual_interference = B_baseline - B_loaded`.
3. **Build curve**: The resulting CSV maps `tcp_payload_gbps → actual_interference_gbps`. Schedule generation uses linear interpolation on this curve to find the right TCP rate for any desired interference level.

Each sweep level uses 8 parallel UCX ports (one per stream) to avoid single-connection TCP throughput limits and TCP TIME_WAIT collisions. Total calibration time is ~2 minutes (16 levels).

The calibration CSV can be saved and reused across runs with `--calibration-csv <path>`.

### Two-node (`run_interfere.sh`)

```
Head node (local)                     Peer node (remote)
─────────────────                     ──────────────────
  [calibration sweep via ib_write_bw]
interfere.py --role client            interfere.py --role server
  ├─ load trace → per-window rates      └─ launch N ucx_receivers
  ├─ apply calibration curve                  ↑ UCX tag recv (TCP)
  ├─ write V2 schedule (rate/N)               │ N parallel streams
  └─ launch N ucx_senders ────────────────→   │ tracks per-1ms BW
       token-bucket paced sends               │ writes CSV
       rate updated per window
```

### Ring topology (`run_ring.sh`)

Each node runs N `ucx_ring_node` processes (default 8) that
both send to the next node and receive from the previous node:

```
        ┌──── ucx_ring_node ────┐     ┌──── ucx_ring_node ────┐
sgpu0   │ send→sgpu2  recv←sgpu4│  -> │ send→sgpu3  recv←sgpu0│  sgpu2
        └───────────────────────┘     └───────────────────────┘
                          ↑                  │
                          │                  ↓
        ┌───────────────────────┐     ┌───────────────────────┐
sgpu4   │ send→sgpu0  recv←sgpu3│  <- │ send→sgpu4  recv←sgpu2│  sgpu3
        └──── ucx_ring_node ────┘     └──── ucx_ring_node ────┘

Calibration between nodes[0]↔nodes[1], then schedule generated
with calibrated rates (rate/N per stream). SCP'd to all nodes.
N processes per node for TCP throughput saturation.
Health check + ib_write_bw monitoring every 30s.
```

### Transport notes

- Both modes use `UCX_TLS=tcp` — kernel TCP handles retransmission and flow control, stable across all network configurations.
- **Multiple parallel streams** (default 8) are used per link to overcome single-TCP-connection throughput limits. Each stream runs a separate `ucx_sender`/`ucx_receiver` pair on consecutive ports, carrying `1/N` of the total target rate. TCP socket buffers are tuned to 4MB with `UCX_TCP_SNDBUF`/`UCX_TCP_RCVBUF`.
- TCP interference competes for link bandwidth with RDMA workloads; the calibration sweep ensures the actual interference matches the desired level despite TCP overhead.
- The calibration uses `ib_write_bw` (RDMA verbs) as a victim workload, which accurately models RDMA-based workloads like NCCL.

## Schedule File Format

| Field | Size | Description |
|-------|------|-------------|
| Magic | 4B | `"SCHD"` |
| Version | 4B | `2` |
| num_windows | 4B | Number of windows |
| msg_size | 4B | Bytes per message |
| window_ns | 8B | Window duration in ns |
| Pad | 8B | Reserved |
| Body | N×8B | `double rate_bps` per window |

## Citation

Noise profiles from:

> De Sensi et al., "Noise in the Clouds: Influence of Network Performance Variability on Application Scalability." POMACS 6(3), 2022. SIGMETRICS 2023. arXiv:2210.15315
