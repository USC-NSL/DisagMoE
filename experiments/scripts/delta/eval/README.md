# AsyncMoE EP16 Final Evaluation — NCSA Delta

## File layout

```
eval/
  ep16_eval.sh      # main: experiment matrix + orchestration loop
  config.sh         # all fixed variables (sourced by main)
  helpers/
    ray.sh          # restart_ray(), stop_ray()
    server.sh       # launch_server(), wait_for_server(), kill_server(), is_oom()
    benchmark.sh    # run_benchmark()
  README.md
```

`helpers/` scripts only define functions; they are sourced, not executed.
To change any single concern, edit only that one file.

---

## What the main script does

For each of the 4 experiments `{sharegpt, legal-court} × {regular, balanced}`,
up to `MAX_RETRIES=3` times:

1. **`restart_ray`** — kills existing server + srun worker steps, stops Ray
   everywhere, restarts head on the current node and workers on the remaining
   SLURM nodes via `srun --overlap`. Required between runs to release GPUs.
2. **`launch_server`** — builds the server command, saves it to
   `server_cmd.sh`, then starts `benchmark/server.py` in the
   background (`nohup`), logging to `server.log`.
3. **`wait_for_server`** — polls the log for `Running on http://0.0.0.0:6699`,
   up to `SERVER_READY_TIMEOUT=1200s` (NFS import on Delta can be slow).
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
| Cluster | 4 nodes × 4 A100-SXM4-40GB = EP16 |
| Placement | `colocate`, dp=16, ep=16 |
| Transport | ZMQ, `--host-ifname hsn0` |
| Scheduler | `defrag` (decay=0.8, lookahead=4, lookback=4) |
| Optimizations | `--cuda-graph-attn --cuda-graph-expert --less-than-sm90` |
| Initial memory fraction | 0.98 |
| OOM step | −0.02 per retry |
| Batch sizes | attn=256, expert=1024 |
| Benchmark | 2000 rps × 5s = 10k reqs, dataset generator (sharegpt), max context len 2048, in/out 256–512 fallback (env-overridable) |
| Advanced logging | disabled |

---

## Placeholders to fill in (`ep16_eval.sh`)

Set the four gate profile paths in the `EXPERIMENTS` array:

```bash
EXPERIMENTS=(
    ".../gating_gptoss120b_sharegpt_200.parquet:sharegpt_regular"
    ".../balanced_output/balanced_gptoss120b_sharegpt_200.parquet:sharegpt_balanced"
    ".../gating_legal_court_opinions_200.parquet:legal_court_regular"
    ".../balanced_output/balanced_legal_court_opinions_200.parquet:legal_court_balanced"
)
```

---

## How to run

```bash
# 1. Shell on head node (inside SLURM allocation)
srun --jobid=<JOBID> --nodelist=<HEAD_NODE> --overlap --pty bash

# 2. Source environment
source ~/DisagMoE/experiments/scripts/delta/env.sh

# 3. Run (RESULTS_DIR is required as the first argument)
cd ~/DisagMoE
bash experiments/scripts/delta/eval/ep16_eval.sh /path/to/my_results \
    |& tee experiments/amoe-081/ep16_eval.log

# Optional: override benchmark parameters via environment
BENCH_RATE=500 BENCH_TIME=10 \
    bash experiments/scripts/delta/eval/ep16_eval.sh /path/to/my_results
```

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
  asyncmoe-legal_court_regular/        ...
  asyncmoe-legal_court_balanced/       ...
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
