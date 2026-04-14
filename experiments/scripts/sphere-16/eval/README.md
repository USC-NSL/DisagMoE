# AsyncMoE EP16 gptoss Evaluation — Sphere-16

## sphere cluster Sphere-16 Cluster Setup & Launch Notes
Cluster: 8 nodes × 2 L40S GPUs = 16 GPUs total
Head node: sgpu2 (10.0.0.1)
Workers:   sgpu3 (10.0.0.2), sgpu4 (10.0.0.3), sgpu5 (10.0.0.4),
           sgpu6 (10.0.0.5), sgpu7 (10.0.0.6), sgpu8 (10.0.0.7),
           sgpu9 (10.0.0.8)
Network:   RoCE via ens1f1np1 (mlx5_1 HCA)
Conda env: disag12 (Python 3.12)
Note: there is NO NFS on sphere cluster, every node is fully bare-metal. You can ssh into every worker to sync files if you change some code.

## File layout

```
eval-gptoss/
  ep16_eval.sh      # main: experiment matrix + orchestration loop
  config.sh         # all fixed variables (sourced by main)
  helpers/
    ray.sh          # restart_ray(), stop_ray()
    server.sh       # launch_server(), wait_for_server(), kill_server(), is_oom()
    benchmark.sh    # run_benchmark()
  README.md
```

`helpers/` scripts only define functions; they are sourced, not executed.

---

## What the main script does

For each of the 4 experiments `{sharegpt, gsm8k} × {regular, balanced}`,
up to `MAX_RETRIES=3` times:

1. **`restart_ray`** — stops Ray on all nodes via SSH, restarts head on
   sgpu0 and workers on sgpu2–9.
2. **`launch_server`** — builds the server command, saves it to
   `server_cmd.sh`, then starts `benchmark/server.py` in the
   background (`nohup`), logging to `server.log`.
3. **`wait_for_server`** — polls the log for `Running on http://0.0.0.0:6699`,
   up to `SERVER_READY_TIMEOUT=300s`.
   - If the server exits early, calls **`is_oom`** on its log. On OOM,
     `MEM_FRAC` is decreased by `MEM_FRAC_STEP=0.02` before the next attempt.
4. **`run_benchmark`** — saves the exact `curl` command to
   `bench_cmd.sh`, then POSTs to `/run_once`; saves response JSON to
   `result.json`.

Final cleanup: `kill_server` + `stop_ray`.

---

## Fixed config (edit `config.sh`)

| Parameter | Value |
|---|---|
| Model | `gptoss_120b` (36 layers, 128 experts, top-4, bf16) |
| Cluster | 8 nodes × 2 L40S = EP16 |
| Head node | sgpu0 (10.0.0.1) |
| Workers | sgpu2, sgpu3, sgpu4, sgpu6, sgpu7, sgpu8, sgpu9 |
| Placement | `colocate`, dp=16, ep=16 |
| Transport | ZMQ, `--host-ifname ens1f1np1`, RoCE via mlx5_1 |
| Scheduler | `defrag` (decay=0.8, lookahead=4, lookback=4) |
| Optimizations | `--cuda-graph-attn --cuda-graph-expert --less-than-sm90` |
| Initial memory fraction | 0.98 |
| OOM step | −0.02 per retry |
| Batch sizes | attn=256, expert=1024 |
| Benchmark | 2000 rps × 5s = 10k reqs, dataset generator (sharegpt), max context len 2048, in/out 256–512 fallback |

---

## IMPORTANT: Gate profile vs. sequence length profile

The gate profile and the benchmark sequence length distribution are **independent configs**.
Changing the gate profile (e.g. from `sharegpt` to `gsm8k`) only changes the expert routing
pattern — it does NOT automatically change the input/output sequence length distribution.

By default, all experiments use `BENCH_DATASET_PATH` (set to `sharegpt_lengths.npy`) and
fixed bounds `in=256–512, out=256–512`. To use gsm8k-representative sequence lengths, you
must override per-workload:

```bash
BENCH_DATASET_PATH=$REPO_DIR/datasets/gsm8k_lengths.npy \
    bash experiments/scripts/sphere-16/eval/gptoss_eval.sh /path/to/results --only gsm8k
```

If you forget this step, gsm8k and sharegpt experiments will produce nearly identical
throughput/latency — because the actual workload (sequence lengths) is the same.

**The SGLang eval scripts handle this automatically** via `BENCH_DATASET_PATHS` + `npy`
dataset mode. The AsyncMoE eval scripts currently do not auto-resolve per-workload — you
must override manually or update the script.

---

## Placeholders to fill in (`ep16_eval.sh`)

Set the four gate profile paths in the `EXPERIMENTS` array:

```bash
EXPERIMENTS=(
    ".../gating_gptoss120b_sharegpt_200.parquet:sharegpt_regular"
    ".../balanced_output/balanced_gptoss120b_sharegpt_200.parquet:sharegpt_balanced"
    ".../gating_math_gsm8k_200.parquet:gsm8k_regular"
    ".../balanced_output/balanced_math_gsm8k_200.parquet:gsm8k_balanced"
)
```

---

## How to run

```bash
# 1. SSH to sgpu0
ssh sgpu0

# 2. Activate conda
conda activate disag12

# 3. Run all experiments (RESULTS_DIR is required as the first argument)
cd ~/DisagMoE
bash experiments/scripts/sphere-16/eval/gptoss_eval.sh /path/to/my_results \
    |& tee /path/to/my_results/ep16_eval.log
```

### Running a single experiment

Use `--list` to see available experiments and `--only` to select which to run:

```bash
# List available experiments (prints index + name, then exits)
bash experiments/scripts/sphere-16/eval/gptoss_eval.sh /path/to/results --list

# Run by index (1-based)
bash experiments/scripts/sphere-16/eval/gptoss_eval.sh /path/to/results --only 1

# Run multiple by index
bash experiments/scripts/sphere-16/eval/gptoss_eval.sh /path/to/results --only 1,3

# Run by name substring (matches against run name, e.g. "asyncmoe-sharegpt_regular")
bash experiments/scripts/sphere-16/eval/gptoss_eval.sh /path/to/results --only sharegpt

# Run one exact experiment
bash experiments/scripts/sphere-16/eval/gptoss_eval.sh /path/to/results --only sharegpt_regular
```

The `--only` filter accepts comma-separated values. Each value is matched as
a 1-based index (if numeric) or as a substring of the run name (e.g.
`asyncmoe-sharegpt_regular`). Omitting `--only` runs all experiments.

---

## Output layout

Run directories are named `<system>-<dataset>` under `RESULTS_DIR`.

```
<RESULTS_DIR>/
  asyncmoe-sharegpt_regular/
    server_cmd.sh                       # exact server launch command (replayable)
    server.log                          # server stdout/stderr
    bench_cmd.sh                        # exact curl command (replayable)
    result.json                         # benchmark response JSON
  asyncmoe-sharegpt_balanced/          ...
  asyncmoe-gsm8k_regular/              ...
  asyncmoe-gsm8k_balanced/             ...
```

On retries (e.g. OOM), failed-attempt artifacts are preserved under `attempt<N>/`; the final successful attempt remains at the top level.

To plot global batch size (running/waiting) timelines: `python experiments/scripts/sphere-16/eval/plot_asyncmoe_inflight_timeline.py <RESULTS_DIR>`
