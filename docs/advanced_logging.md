# Advanced Logging for MoE Diagnostics

Low-overhead, random-sampled instrumentation for DisagMoE's MoE execution pipeline. Captures per-GPU batch sizes, groupedGEMM timings, per-(layer, expert) queuing delays, and per-schedule-step queue depth snapshots.

## What It Captures

1. **MoE Step Batch Sizes** — total tokens per groupedGEMM call (sampled)
2. **MoE Step Execution Times** — wall-clock milliseconds per MoE step, including CUDA sync (sampled)
3. **MoE Step Timestamps** — `time.monotonic()` for each sampled step (seconds, per-process)
4. **Queuing Delays** — per (layer, expert) scheduling delay in ms, with timestamps (sampled)
5. **Queue Depth Snapshots** — full queue-depth vector at every schedule decision, atomically aligned with the scheduled layer ID (not sampled — logged on every non-empty batch)

## Enabling

### CLI flags

```
--enable-advanced-logging              # default: off
--advanced-logging-dir ./advanced_logs # output directory
--advanced-logging-sample-rate 0.1     # 0.0–1.0, default 10% (note: this sampling is NOT applied to snapshot&sched logs)
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

Logs are dumped at the end of each benchmark run. Each GPU worker writes to its own subdirectory:

```
advanced_logs/
├── device_0/
│   ├── moe_steps.json
│   ├── queuing_delays.json
│   └── queue_snapshots.json
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

Each entry corresponds to one **sampled** MoE forward pass (groupedGEMM w13 + activation + w2). The timing includes a CUDA stream sync on the sampled step. Timestamps are `time.monotonic()` values — use differences for chronological analysis (absolute values are per-process).

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

## Dump Flow

1. **Worker-local dump**: each Ray worker calls `AdvancedLogger.dump()`, which writes the three JSON files to the worker's local filesystem under `<advanced_logging_dir>/device_<device_id>/`
2. **Controller SCP gather**: the controller calls `scp -rq` from each remote worker node to the head node, assembling all device subdirectories into the final output directory
3. **Trigger**: dump happens automatically at the end of a benchmark run, or manually via the HTTP endpoint

## Manual Log Dump (API Server Mode)

```bash
curl -X POST http://localhost:6699/dump_advanced_logs \
  -H "Content-Type: application/json" \
  -d '{"suffix": "_run1"}'
```

When a suffix is provided, output files are named `moe_steps_run1.json`, etc.

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
- `heatmap_queue_per_expert.png` — layer × expert queuing delay heatmap
- `heatmap_queue_per_rank.png` — layer × rank (experts averaged) queuing delay heatmap
- `rank_queue_timeseries/` — per-rank full-run queue depth timeseries (interleaved attn/expert rows)
- `rank_queue_timeseries_mid10s/` — same, zoomed to middle 10 seconds
- `rank_queue_timeseries_mid1s/` — same, zoomed to middle 1 second

## Design Notes

- **Sampling**: `moe_steps` and `queuing_delays` are sampled at the configured rate (default 10%). `queue_snapshots` are **not** sampled — every schedule decision is logged.
- **CUDA sync cost**: `torch.cuda.synchronize()` is called only for the MoE execution time measurement, and only when that step is sampled. This adds ~0.1ms per sampled step.
- **No C++ changes**: all instrumentation is pure Python. The C++ scheduler exposes `schedule_trace()` which returns the batch and queue snapshot atomically, but no additional C++ instrumentation was added.
- **Thread safety**: each GPU worker has its own `AdvancedLogger` instance — no sharing.
- **Multi-node collection**: workers dump to their local disk; the controller uses SCP to gather everything to the head node. This avoids serializing large JSON blobs through Ray object store.
- **Reset**: `AdvancedLogger.reset()` clears all accumulated data, used between benchmark runs.
