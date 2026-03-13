# Advanced Logging for MoE Diagnostics

Low-overhead, random-sampled instrumentation for DisagMoE's MoE execution pipeline.

## What It Captures

1. **MoE Batch Size CDF** — Total tokens per groupedGEMM call (sampled)
2. **MoE Execution Time CDF** — Wall-clock milliseconds per MoE step (sampled, includes CUDA sync)
3. **MoE Step Timestamps** — `time.monotonic()` timestamp (seconds) for each sampled step, enabling chronological analysis
4. **Queuing Delay Heatmap** — Per (layer × expert) average scheduling delay in ms (sampled)

## Enabling

In `benchmark/scripts/launch_server.sh`, set:

```bash
ENABLE_ADVANCED_LOGGING=1
ADVANCED_LOGGING_DIR="./advanced_logs"
ADVANCED_LOGGING_SAMPLE_RATE=0.1  # 0.0–1.0, default 10%
```

When `ENABLE_ADVANCED_LOGGING=0` (default), the system bears **zero overhead** — all logging calls short-circuit on a boolean check without any timing, allocation, or I/O.

The sample rate can also be set via CLI: `--advanced-logging-sample-rate 0.2`.

## Output Format

Logs are dumped automatically at the end of each benchmark run. Each GPU worker writes to its own subdirectory:

```
advanced_logs/
├── device_0/
│   ├── moe_steps.json
│   └── queuing_delays.json
├── device_1/
│   ├── moe_steps.json
│   └── queuing_delays.json
...
```

### `moe_steps.json`

```json
{
  "batch_sizes": [128, 256, 64, ...],
  "execution_times_ms": [2.31, 4.57, 1.12, ...],
  "timestamps_s": [1234.567, 1234.891, 1235.003, ...]
}
```

Each entry corresponds to one sampled MoE forward pass (groupedGEMM w13 + activation + w2). Timestamps are `time.monotonic()` values in seconds — use differences for chronological analysis (absolute values are per-process).

### `queuing_delays.json`

```json
{
  "5_12": {
    "layer_id": 5,
    "expert_id": 12,
    "delays_ms": [0.45, 0.32, ...],
    "mean_ms": 0.38,
    "count": 47
  },
  ...
}
```

Keys are `"{layer_id}_{expert_id}"`. Each delay is the per-token scheduling time (total schedule time / batch size) for that (layer, expert) pair.

## Manual Log Dump (API Server Mode)

When running the Flask API server, you can trigger a log dump via:

```bash
curl -X POST http://localhost:6699/dump_advanced_logs \
  -H "Content-Type: application/json" \
  -d '{"suffix": "_run1"}'
```

## Plotting

```bash
# Generate CDF plots + queuing heatmap
python plots/plot_advanced_logs.py ./advanced_logs --label "ShareGPT r500"

# Specify output directory
python plots/plot_advanced_logs.py ./advanced_logs --label "GSM8K r500" --out-dir ./plots
```

This produces:
- `moe_cdf_{label}.png` — 1×2 subplot with batch size CDF and execution time CDF
- `queuing_heatmap_{label}.png` — Layer × Expert heatmap of mean queuing delays

## Design Notes

- **Sampling rate**: 10% by default. Configurable via `ADVANCED_LOGGING_SAMPLE_RATE` in `launch_server.sh` or `--advanced-logging-sample-rate` CLI arg.
- **CUDA sync cost**: `torch.cuda.current_stream().synchronize()` is called only for the MoE execution time measurement, and only when that step is sampled (~10%). This adds ~0.1ms per sampled step.
- **No C++ changes**: All instrumentation is pure Python, no recompilation needed.
- **Thread safety**: Each GPU worker has its own `AdvancedLogger` instance — no sharing.
