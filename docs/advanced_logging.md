# Advanced Logging for MoE Diagnostics

Low-overhead instrumentation for DisagMoE's MoE execution pipeline and cross-rank transport path. It captures per-GPU groupedGEMM behavior, scheduler queuing state, dispatcher/pool activity, and corrected receive-side / backpressure signals for multi-node bottleneck analysis.

## What It Captures

1. **MoE Step Batch Sizes** — total tokens per groupedGEMM call (sampled)
2. **MoE Step Execution Times** — wall-clock milliseconds per MoE step, including CUDA sync (sampled)
3. **MoE Step Timestamps** — wall-clock timestamps for each sampled step (seconds, per-process)
4. **Queuing Delays** — per (layer, expert) scheduling delay in ms, with timestamps (sampled)
5. **Queue Depth Snapshots** — full queue-depth vector at every schedule decision, atomically aligned with the scheduled layer ID (not sampled — logged on every non-empty batch)
6. **Dispatcher Put Calls** — sampled `dispatcher.put()` metadata: batch size, latency, layer id, timestamp
7. **Pool Admissions** — timestamps for new requests entering the frontend pool
8. **Receive Completions** — recv-side transport completions with peer id, layer id, bytes, tokens, posted time, completed time, and local-vs-remote flag
9. **Pending-Send Stall Events** — explicit wait intervals when `max_pending_sends` throttles the dispatcher

## Enabling

### CLI flags

```
--enable-advanced-logging              # default: off
--advanced-logging-dir ./advanced_logs # output directory
--advanced-logging-sample-rate 0.1     # 0.0–1.0, default 10% (sampling is NOT applied to queue snapshots or transport-completion / stall logs)
```

### Launch script

In `experiments/scripts/sphere-16/gptoss/launch_server.sh`:

```bash
ENABLE_ADVANCED_LOGGING=1
ADVANCED_LOGGING_DIR="./advanced_logs"
ADVANCED_LOGGING_SAMPLE_RATE=0.1
```

When `ENABLE_ADVANCED_LOGGING=0` (default), all logging calls short-circuit on a boolean check — **zero overhead**.

## Output Format

Logs are dumped at the end of each benchmark run, or manually via the dump endpoint. Each GPU worker writes to its own subdirectory:

```
advanced_logs/
├── device_0/
│   ├── moe_steps.json
│   ├── queuing_delays.json
│   ├── queue_snapshots.json
│   ├── dispatcher_puts.json
│   ├── pool_puts.json
│   ├── recv_completions.json
│   └── pending_send_stalls.json
├── device_1/
│   ├── moe_steps.json
│   ├── queuing_delays.json
│   └── queue_snapshots.json
...
└── device_15/
    └── ...
```

### `moe_steps.json`

```json
{
  "batch_sizes": [128, 256, 64, ...],
  "execution_times_ms": [2.31, 4.57, 1.12, ...],
  "timestamps_s": [164159.287, 164159.318, 164159.388, ...]
}
```

Each entry corresponds to one **sampled** MoE forward pass (groupedGEMM w13 + activation + w2). The timing includes a CUDA stream sync on the sampled step. Timestamps are wall-clock values captured at log time; use relative differences for chronology.

### `queuing_delays.json`

```json
{
  "30_0": {
    "layer_id": 30,
    "expert_id": 0,
    "delays_ms": [0.003, 0.0026, ...],
    "timestamps_s": [164160.123, 164160.456, ...],
    "mean_ms": 0.0028,
    "count": 47
  },
  ...
}
```

Keys are `"{layer_id}_{expert_id}"`. Each delay is the per-token scheduling time for that (layer, expert) pair. Only logged on **sampled** expert batches.

### `queue_snapshots.json`

```json
{
  "timestamps_s": [164159.001, 164159.002, ...],
  "scheduled_layer_ids": [0, 37, 1, 2, ...],
  "layer_depths": [[1, 0, 0, ...], [0, 3, 0, ...], ...]
}
```

Logged on **every** non-empty schedule step (not subject to sampling).

- `timestamps_s` — `time.monotonic()` at the schedule decision
- `scheduled_layer_ids` — the unified layer index chosen by the scheduler. Attention layers use their `layer_id` directly; expert layers use `layer_id + num_attn_layers_in_pool`
- `layer_depths` — a flat vector of queue depths for all layers managed by this worker, captured **atomically with** the schedule decision via `schedule_trace()` in the C++ scheduler. The vector layout is `[attn_0, attn_1, ..., attn_N, expert_0, expert_1, ..., expert_M]`

The atomic alignment between `scheduled_layer_ids[i]` and `layer_depths[i]` means each snapshot shows the exact queue state the scheduler saw when it made that decision.

### `dispatcher_puts.json`

```json
{
  "num_tokens": [64, 128, 32, ...],
  "latencies_ms": [0.003, 0.002, 0.004, ...],
  "layer_ids": [17, 17, 28, ...],
  "timestamps_s": [171312.100, 171312.101, 171312.140, ...]
}
```

These are sampled frontend-side dispatcher call records. They are useful for qualitative visibility, but **not** a correct proxy for network bandwidth because dispatch is asynchronous.

### `pool_puts.json`

```json
{
  "num_tokens": [1, 1, 1, ...],
  "timestamps_s": [171312.050, 171312.051, 171312.052, ...]
}
```

Tracks request admission into the frontend pool.

### `recv_completions.json`

```json
{
  "peer_ids": [3, 8, 4, ...],
  "layer_ids": [12, 29, 30, ...],
  "num_tokens": [64, 128, 96, ...],
  "num_bytes": [524288, 1048576, 786432, ...],
  "posted_timestamps_s": [171312.200, 171312.201, ...],
  "completed_timestamps_s": [171312.201, 171312.203, ...],
  "is_local": [false, false, true, ...]
}
```

This is the corrected transport signal. Each entry records when a recv was posted and when completion was actually observed by the receiver. For local transfers, `is_local=true`; for NIC traffic, `is_local=false`.

### `pending_send_stalls.json`

```json
{
  "start_timestamps_s": [171312.240, 171312.244, ...],
  "end_timestamps_s": [171312.241, 171312.245, ...],
  "pending_before": [16, 16, ...],
  "max_pending": [16, 16, ...],
  "yield_counts": [102, 88, ...]
}
```

These entries are emitted when the dispatcher is blocked in `drain_pending_sends_to(max_pending_sends)` because in-flight async sends have not retired yet. This is the correct way to observe `max_pending_sends` backpressure.

## Dump Flow

1. **Worker-local dump**: each Ray worker calls `AdvancedLogger.dump()`, which writes all JSON files to the worker's local filesystem under `<advanced_logging_dir>/device_<device_id>/`
2. **Controller gather**: the controller gathers remote worker outputs onto the head node (historically via SCP/rsync-based collection)
3. **Trigger**: dump happens automatically at the end of a benchmark run, or manually via the HTTP endpoint

If a run is terminated early or a remote gather is interrupted, some per-device JSON files may be truncated. The plotting script tolerates corrupt JSON by skipping those files with a warning.

## Manual Log Dump (API Server Mode)

```bash
curl -X POST http://localhost:6699/dump_advanced_logs \
  -H "Content-Type: application/json" \
  -d '{"suffix": "_run1"}'
```

When a suffix is provided, output files are named `moe_steps_run1.json`, etc.

For long-running experiments or when benchmarking cleanup is slow, it is valid to trigger a manual mid-run dump to capture a steady-state snapshot.

## Plotting

```bash
python experiments/process_and_plot_advanced_logging.py <advanced_logs_dir> [output_dir]
```

Produces:
- `summary.txt` — per-rank and aggregate statistics (batch size, execution time: mean/p50/p99/max)
- `cdf_gemm_time.png` — per-rank CDF of groupedGEMM execution times
- `cdf_gemm_batchsize.png` — per-rank CDF of groupedGEMM batch sizes
- `bsz_vs_time.png` — per-batch-size mean execution time (per rank)
- `bsz_vs_time_avg.png` — same, averaged across all ranks with ±1 std band
- `cdf_gemm_time_20_40s.png` — per-rank groupedGEMM time CDF restricted to 20–40s into the run
- `cdf_gemm_batchsize_20_40s.png` — per-rank groupedGEMM batch-size CDF restricted to 20–40s into the run
- `bsz_vs_time_20_40s.png` — per-rank groupedGEMM batch-size vs compute-time curve restricted to 20–40s
- `bsz_vs_time_avg_20_40s.png` — all-rank averaged version of the same 20–40s peak window
- `heatmap_queue_per_expert.png` — layer × expert queuing delay heatmap
- `heatmap_queue_per_rank.png` — layer × rank (experts averaged) queuing delay heatmap
- `rank_queue_timeseries/` — per-rank full-run queue depth timeseries (interleaved attn/expert rows)
- `rank_queue_timeseries_mid10s/` — same, zoomed to middle 10 seconds
- `rank_queue_timeseries_mid1s/` — same, zoomed to middle 1 second
- `global_running_requests.png` — global DP scheduler timeline parsed from `server.log` (`#running requests` and `#waiting requests`)
- `dispatcher_pool_summary.txt` — sampled dispatcher/pool summary (if those logs exist)
- `recv_bandwidth_summary.txt` — recv-side bandwidth and pending-send stall summary (if corrected transport logs exist)
- `node_receive_bandwidth.png` — per-node receive bandwidth timeline, aggregating the 2 ranks that share one NIC
- `pending_send_stalls.png` — scatter plot of explicit `max_pending_sends` stall events

The plotting script automatically looks for `server.log` one directory above `<advanced_logs_dir>` to generate `global_running_requests.png`.

## Design Notes

- **Sampling**: `moe_steps`, `queuing_delays`, and dispatcher/pool frontend logs may be sampled at the configured rate. `queue_snapshots` are **not** sampled. Receive-completion and pending-send-stall logs are emitted whenever those events occur.
- **CUDA sync cost**: `torch.cuda.synchronize()` is called only for the MoE execution time measurement, and only when that step is sampled. This adds ~0.1ms per sampled step.
- **Correct transport methodology**: `dispatcher.put()` latency is **not** a valid network-bandwidth metric because the actual transport is asynchronous. Use `recv_completions.json` and `pending_send_stalls.json` for bandwidth / backpressure analysis.
- **C++ instrumentation now exists**: queue snapshots still rely on C++ `schedule_trace()`, and corrected transport logging adds C++ instrumentation in the dispatcher / pool path.
- **Thread safety**: each GPU worker has its own `AdvancedLogger` instance — no sharing.
- **Multi-node collection**: workers dump to their local disk; the controller uses SCP to gather everything to the head node. This avoids serializing large JSON blobs through Ray object store.
- **Reset**: `AdvancedLogger.reset()` clears all accumulated data, used between benchmark runs.
