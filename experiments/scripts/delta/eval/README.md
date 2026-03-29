# AsyncMoE EP16 Evaluation — NCSA Delta

## File layout

```
eval/
  gptoss_eval.sh       # main: gptoss experiment matrix + orchestration loop
  glm45air_eval.sh     # main: glm45air experiment matrix + orchestration loop
  config.sh            # shared cluster/runtime/benchmark config (sourced by model configs)
  gptoss_config.sh     # gptoss_120b model config (sources config.sh)
  glm45air_config.sh   # glm45air_106b model config (sources config.sh)
  ep16_eval.sh         # (legacy) original single-model eval script
  helpers/
    ray.sh             # restart_ray(), stop_ray()
    server.sh          # launch_server(), wait_for_server(), kill_server(), is_oom()
    benchmark.sh       # run_benchmark()
  README.md
```

`helpers/` scripts only define functions; they are sourced, not executed.
To change any single concern, edit only that one file.

---

## What the main scripts do

For each experiment in the matrix (e.g. `{sharegpt, gsm8k} × {regular, balanced}`),
up to `MAX_RETRIES=3` times:

1. **`restart_ray`** — kills existing server + srun worker steps, stops Ray
   everywhere, restarts head on the current node and workers on the remaining
   SLURM nodes via `srun --overlap`. Required between runs to release GPUs.
2. **`launch_server`** — builds the server command, saves it to
   `server_cmd.sh`, then starts `benchmark/server.py` in the
   background (`nohup`), logging to `server.log`.
3. **`wait_for_server`** — polls the log for `Running on http://0.0.0.0:6699`,
   up to `SERVER_READY_TIMEOUT=600s`.
   - If the server exits early, calls **`is_oom`** on its log. On OOM,
     `MEM_FRAC` is decreased by `MEM_FRAC_STEP=0.02` before the next attempt.
   - OOM is also checked when the benchmark itself fails (runtime OOM).
4. **`run_benchmark`** — saves the exact `curl` command to
   `bench_cmd.sh`, then POSTs to `/run_once`; saves response JSON to
   `result.json`.

Final cleanup: `kill_server` + `stop_ray`.

---

## Config architecture

- **`config.sh`** — shared cluster, runtime, scheduler, and benchmark settings
- **`gptoss_config.sh`** — sources `config.sh`, then sets gptoss_120b model vars
- **`glm45air_config.sh`** — sources `config.sh`, then sets glm45air_106b model vars (includes shared expert config)

Each eval script sources its model-specific config, which in turn sources the shared config.

---

## Fixed config (edit `config.sh`)

| Parameter | Value |
|---|---|
| Cluster | 4 nodes × 4 A100-SXM4-40GB = EP16 |
| Placement | `colocate`, dp=16, ep=16 |
| Transport | ZMQ, `--host-ifname hsn0` |
| Scheduler | `defrag` (decay=0.8, lookahead=4, lookback=4) |
| Optimizations | `--cuda-graph-attn --cuda-graph-expert --less-than-sm90` |
| Initial memory fraction | 0.92 |
| OOM step | −0.02 per retry |
| Batch sizes | attn=256, expert=1024 |
| Benchmark | 2000 rps × 5s = 10k reqs, dataset auto-selected per experiment (sharegpt or gsm8k), max context len 2048, in/out 256–512 fallback (env-overridable) |
| Benchmark timeout | sharegpt: 600s, gsm8k: 300s (auto-selected from dataset path) |
| Server ready timeout | 600s |
| Throughput analysis window | 15–60s |

---

## How to run

```bash
# 1. Shell on head node (inside SLURM allocation)
srun --jobid=<JOBID> --nodelist=<HEAD_NODE> --overlap --pty bash

# 2. Source environment (conda must be initialized first)
source ~/miniconda3/etc/profile.d/conda.sh
source ~/DisagMoE/experiments/scripts/delta/env.sh

# 3. Kill any lingering processes from a previous run
#    (kill $PID only removes the nohup wrapper — children must be killed explicitly)
ps aux | grep -E 'ray|server\.py|benchmark' | grep -v grep
pkill -f "benchmark/server.py" 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
~/miniconda3/envs/amoe/bin/ray stop --force 2>/dev/null || true

# 4. Launch gptoss in background
#    If running from inside an srun --pty shell, SLURM_JOB_ID / SLURM_JOB_NODELIST
#    are already set. HEAD_IP is auto-detected via hostname -I. Only set them
#    explicitly when launching from a non-SLURM shell (e.g. automated runner).
OUTDIR=~/unified-eval-mar28-1
cd ~/DisagMoE
SLURM_JOB_ID=<JOBID> SLURM_JOB_NODELIST=<NODELIST> HEAD_IP=<HEAD_NODE_IP> \
    nohup bash experiments/scripts/delta/eval/gptoss_eval.sh "$OUTDIR/asyncmoe-gptoss" \
    > "$OUTDIR/asyncmoe-gptoss.log" 2>&1 &
echo $! > "$OUTDIR/asyncmoe-gptoss.pid"

# 5. After gptoss finishes, launch glm45air
SLURM_JOB_ID=<JOBID> SLURM_JOB_NODELIST=<NODELIST> HEAD_IP=<HEAD_NODE_IP> \
    nohup bash experiments/scripts/delta/eval/glm45air_eval.sh "$OUTDIR/asyncmoe-glm45air" \
    > "$OUTDIR/asyncmoe-glm45air.log" 2>&1 &
echo $! > "$OUTDIR/asyncmoe-glm45air.pid"
```

### Running a single experiment

Use `--list` to see available experiments and `--only` to select which to run:

```bash
# List available experiments (prints index + name, then exits)
bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results --list

# Run by index (1-based)
bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results --only 1

# Run multiple by index
bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results --only 1,3

# Run by name substring (matches against run name, e.g. "asyncmoe-sharegpt_regular")
bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results --only sharegpt

# Run one exact experiment
bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results --only sharegpt_regular
```

The `--only` filter accepts comma-separated values. Each value is matched as
a 1-based index (if numeric) or as a substring of the run name (e.g.
`asyncmoe-sharegpt_regular`). Omitting `--only` runs all experiments.

---

## Overriding benchmark dataset and context length

### Benchmark dataset

Each experiment auto-selects its dataset based on the experiment label:
- `sharegpt*` experiments → `datasets/sharegpt_lengths.npy`
- `gsm8k*` experiments → `datasets/gsm8k_lengths.npy`

To force a specific dataset for all experiments in a run, set `BENCH_DATASET_PATH`:

```bash
BENCH_DATASET_PATH=~/DisagMoE/datasets/sharegpt_lengths.npy \
    bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results

BENCH_DATASET_PATH=~/DisagMoE/datasets/gsm8k_lengths.npy \
    bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results
```

### Context length

Default max context length is 2048. Override with `BENCH_MAX_CONTEXT_LEN`:

```bash
BENCH_MAX_CONTEXT_LEN=4096 \
    bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results
```

Both can be combined:

```bash
BENCH_DATASET_PATH=~/DisagMoE/datasets/sharegpt_lengths.npy \
BENCH_MAX_CONTEXT_LEN=4096 \
    bash experiments/scripts/delta/eval/gptoss_eval.sh /path/to/results --only sharegpt
```

Other overridable benchmark parameters (all from `config.sh`): `BENCH_RATE`, `BENCH_TIME`, `BENCH_MIN_IN`, `BENCH_MAX_IN`, `BENCH_MIN_OUT`, `BENCH_MAX_OUT`.

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
  asyncmoe-gsm8k_regular/             ...
  asyncmoe-gsm8k_balanced/            ...
```

On retries (e.g. OOM), failed-attempt artifacts are preserved under `attempt<N>/`;
the final successful attempt remains at the top level.

```
<RESULTS_DIR>/
  asyncmoe-sharegpt_regular/
    attempt1/                           # archived from first (failed) attempt
      server.log
      server_cmd.sh
    server_cmd.sh                       # final successful attempt
    server.log
    bench_cmd.sh
    result.json
```
