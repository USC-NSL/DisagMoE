# Throughput vs ITL Sweep Experiment Guide

## Overview

This experiment measures how **token throughput** and **inter-token latency (ITL)** change as request rate varies, for both AsyncMoE and SGLang EP16 on the gptoss-120b model. Unlike the standard fixed-rate eval (2000 rps × 10s), this sweep sends requests at controlled rates (25–400 rps) for an extended duration (200s) to capture steady-state behavior at different load levels.

**Purpose**: Generate tput-vs-ITL curves that show each system's throughput-latency tradeoff across the operating range.

## Cluster

- **Nodes**: 8 × sgpu (sgpu0 head, sgpu2–sgpu9 workers)
- **GPUs**: 2 × L40S per node = 16 total
- **Network**: RoCE via `ens1f1np1` (`mlx5_1` HCA)
- **No NFS** — every node is bare-metal, code must be synced via rsync

## Prerequisites

### Software on every node

| Component | Path | Notes |
|---|---|---|
| Miniconda | `~/miniconda3` | |
| DisagMoE | `~/DisagMoE` | `disag12` conda env, `make pip` built |
| sglang-fake-prefill | `~/sglang-fake-prefill` | `sglang-fp` conda env |
| gdrcopy | `~/gdrcopy` | kernel module `gdrdrv` must be loaded |
| System packages | apt | `libzmq3-dev libcereal-dev libucx-dev libnccl2 libnccl-dev` |
| ld.so.conf entries | `/etc/ld.so.conf.d/` | conda env lib paths + gdrcopy |

### After every cluster reboot

Reboots wipe the gdrdrv kernel module and may reset ld.so.conf. Run on **every** node:

```bash
# Load gdrdrv
cd ~/gdrcopy && sudo bash ./insmod.sh

# Restore linker paths
sudo tee /etc/ld.so.conf.d/conda-disag12.conf >/dev/null <<EOF
$HOME/miniconda3/envs/disag12/lib
$HOME/miniconda3/envs/disag12/lib/python3.12/site-packages/torch/lib
EOF
sudo tee /etc/ld.so.conf.d/conda-sglang-fp.conf >/dev/null <<EOF
$HOME/miniconda3/envs/sglang-fp/lib
$HOME/miniconda3/envs/sglang-fp/lib/python3.12/site-packages/torch/lib
EOF
sudo tee /etc/ld.so.conf.d/gdrcopy.conf >/dev/null <<EOF
/usr/local/gdrcopy/lib
EOF
sudo ldconfig
```

Verify imports work:
```bash
# AsyncMoE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12
python -c "import disagmoe_c; print('OK')"

# SGLang
conda activate sglang-fp
python -c "import sglang; print('OK')"
```

### Syncing code changes

Since there's no NFS, after ANY code change on sgpu0, rsync to all workers:

```bash
for n in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  rsync -az --delete --exclude='__pycache__' --exclude='.git' \
    --exclude='experiments/' --exclude='build/' --exclude='*.egg-info' \
    ~/DisagMoE/ $n:~/DisagMoE/ &
  rsync -az --delete --exclude='__pycache__' --exclude='.git' \
    --exclude='build/' --exclude='*.egg-info' \
    ~/sglang-fake-prefill/ $n:~/sglang-fake-prefill/ &
done; wait
```

If C++ code changed (csrc/), rebuild with `make pip` on every node.
If only Python changed, rsync alone is sufficient (editable install).

## Experiment Parameters

| Parameter | Value |
|---|---|
| Model | gptoss_120b (36 layers, 128 experts, top-4, bf16) |
| Gate profiles | balanced variants only |
| Sequence lengths | sampled from `.npy` dataset files (sharegpt or gsm8k) |
| Input/output bounds | 256–512 tokens |
| Max context length | 2048 |
| Duration per rate | 200 seconds of sending, then wait for drain |
| Total requests | `rate × 200` |

### Request rates to sweep

| Workload | Rates (rps) |
|---|---|
| sharegpt_balanced | 25, 50, 100, 150, 200, 300 |
| gsm8k_balanced | 25, 50, 100, 150, 200, 300, 400 |

## Running AsyncMoE

AsyncMoE uses the standard eval script with env var overrides for rate and duration.

### Setup

```bash
ssh sgpu0
source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12
cd ~/DisagMoE
```

### Per-run procedure

```bash
# Set variables
RATE=100
PROFILE=sharegpt_balanced   # or gsm8k_balanced
OUTDIR=experiments/sphere-16-tput-itl-mar30/asyncmoe-gptoss-results

# For gsm8k, also set:
# DATASET_OVERRIDE="BENCH_DATASET_PATH=$HOME/DisagMoE/datasets/gsm8k_lengths.npy"

# Launch in tmux with automatic rename
tmux new-session -d -s amoe-run
tmux send-keys -t amoe-run "source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate disag12 && cd ~/DisagMoE && \
  BENCH_RATE=$RATE BENCH_TIME=200 BENCH_CURL_TIMEOUT=3600 \
  ANALYZE_THROUGHPUT_WINDOW=30,190 $DATASET_OVERRIDE \
  bash experiments/scripts/sphere-16/eval/gptoss_eval.sh $OUTDIR \
    --only ${PROFILE} \
  2>&1 | tee experiments/sphere-16-tput-itl-mar30/asyncmoe-${PROFILE}-${RATE}rps.log && \
  mv $OUTDIR/asyncmoe-${PROFILE} $OUTDIR/asyncmoe-${PROFILE}-${RATE}rps && \
  echo 'RENAME DONE'" Enter
```

### Monitoring

- **1 min after launch**: check server started (`grep "Running on" <logfile>`)
- **Every 5 min**: check detokenizer throughput lines in `server.log`
  - Healthy: "token throughput: XX.XXk tokens/s" every ~10 seconds
  - No throughput for >60s = hung → kill and retry
- The eval script handles Ray restart between runs automatically

### Key env var overrides (AsyncMoE config.sh)

| Variable | Default | Override for sweep |
|---|---|---|
| `BENCH_RATE` | 2000 | target rate (e.g. 100) |
| `BENCH_TIME` | 10 | 200 |
| `BENCH_CURL_TIMEOUT` | 1200 | 3600 (1h, for high-request runs) |
| `BENCH_DATASET_PATH` | `datasets/sharegpt_lengths.npy` | `datasets/gsm8k_lengths.npy` for gsm8k |
| `ANALYZE_THROUGHPUT_WINDOW` | 15,45 | 30,190 (wider window for 200s runs) |

## Running SGLang

SGLang is significantly more fragile. Follow this procedure **exactly**.

### Critical differences from AsyncMoE

1. **`--disable-stream` is mandatory** — without it, sustained request rates cause HTTP connection exhaustion (ephemeral port depletion and connection resets)
2. **Ports must be explicitly freed between runs** — SGLang uses ports 25000–25005 and 30000; zombie processes hold them after kills
3. **Run one experiment at a time** — never batch into a sweep script (the output directory naming is fragile)
4. **The eval script writes to a fixed default directory** (`sglang_ep16-<profile>`) — you must rename it after each run with `&& mv`

### Pre-flight (MANDATORY before EVERY run)

```bash
# Kill all sglang processes and free ports on ALL nodes
pkill -9 -f sglang; pkill -9 -f "python -m sglang"
for n in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh $n "pkill -9 -f sglang; pkill -9 -f 'python -m sglang'; \
    tmux kill-server; \
    fuser -k 25000/tcp 25001/tcp 25002/tcp 25003/tcp 25004/tcp 25005/tcp 30000/tcp" &
done; wait; sleep 8

# Verify ports are free
fuser 25000/tcp 25004/tcp 30000/tcp  # must produce no output

# Clean the default output directory
rm -rf ~/DisagMoE/experiments/<exp-dir>/sglang-gptoss-results/sglang_ep16-${PROFILE}
```

### Per-run procedure

```bash
RATE=100
PROMPTS=$((RATE * 200))
PROFILE=gsm8k_balanced   # or sharegpt_balanced
EXPDIR=~/DisagMoE/experiments/sphere-16-tput-itl-mar30

tmux new-session -d -s sgrun
tmux send-keys -t sgrun "source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate sglang-fp && cd ~/sglang-fake-prefill && \
  BENCH_REQUEST_RATE=$RATE BENCH_NUM_PROMPTS=$PROMPTS BENCH_TIMEOUT=3600 \
  BENCH_DISABLE_STREAM=1 MAX_RETRIES=2 \
  bash experiments/sphere/eval/gptoss_eval.sh \
    $EXPDIR/sglang-gptoss-results --only '_ep16-${PROFILE}' \
  2>&1 | tee $EXPDIR/sglang_ep16-${PROFILE}-${RATE}rps.log && \
  mv $EXPDIR/sglang-gptoss-results/sglang_ep16-${PROFILE} \
     $EXPDIR/sglang-gptoss-results/sglang_ep16-${PROFILE}-${RATE}rps && \
  echo 'RENAME DONE'" Enter
```

### Monitoring

- **1 min after launch**: check for startup errors
  ```bash
  grep "exception\|FAILED\|Server ready\|Scheduler hit" <logfile> | tail -5
  ```
  SGLang servers frequently crash in the first 30 seconds. If you see `Scheduler hit an exception` or `Connection closed by peer`, the run is dead — kill and retry.
- **Every 5 min**: check the server head log for throughput
  ```bash
  grep "Throughput:" .../logs/server_head.log | tail -3
  ```

### Key env var overrides (SGLang config.sh)

| Variable | Default | Override for sweep |
|---|---|---|
| `BENCH_REQUEST_RATE` | 2000 | target rate |
| `BENCH_NUM_PROMPTS` | 20000 | `rate × 200` |
| `BENCH_TIMEOUT` | 1500 | 3600 |
| `BENCH_DISABLE_STREAM` | 0 | **1** (mandatory for sustained rates) |

## Known Issues and Pitfalls

### 1. SGLang HTTP connection pressure (CRITICAL)

At sustained rates ≥100 rps with streaming enabled, the benchmark client exhausts ephemeral ports or overwhelms the HTTP server, causing `ConnectionResetError: [Errno 104] Connection reset by peer` or `OSError: [Errno 99] Cannot assign requested address`. Even with `net.ipv4.tcp_tw_reuse=1` and expanded port ranges, this persists at high rates.

**Mitigation**: Always use `BENCH_DISABLE_STREAM=1`. This means SGLang ITL metrics from `bench_result.json` will be zero — use the **server-side detokenizer logs** for ITL data instead.

### 2. SGLang Gloo crash in `prepare_mlp_sync_batch_raw`

Before the cluster reboot, SGLang consistently crashed with:
```
RuntimeError: [gloo/transport/tcp/pair.cc:544] Connection closed by peer [10.0.0.8]:7398
```
at ~500 in-flight requests, in `torch.distributed.all_gather_into_tensor`. This was caused by flaky hardware on the node. **A full cluster reboot resolved this issue.**

### 3. Port retention between SGLang runs

SGLang binds to ports 25000–25005 (distributed init, metrics, scheduler) and 30000 (HTTP). After killing the server, these ports can remain in `TIME_WAIT` for 60+ seconds. The next server launch will fail with `port is already in use`.

**Mitigation**: Always run `fuser -k` on all ports on all nodes as part of the pre-flight procedure. Wait 8 seconds after killing before relaunching.

### 4. After cluster reboot: gdrdrv, ldconfig, conda

Reboots wipe:
- The `gdrdrv` kernel module → `ImportError: cannot open shared object file`
- ld.so.conf entries → `ImportError: libzmq.so.5: cannot open shared object file`
- Any node that was never fully set up (e.g. sgpu7 needed miniconda, gdrcopy, apt packages, and `make pip` from scratch)

Always verify all 8 nodes can `import disagmoe_c` and `import sglang` after a reboot.

### 5. Eval script output directory naming

Both eval scripts write results to a **fixed default directory** (e.g. `asyncmoe-sharegpt_balanced` or `sglang_ep16-sharegpt_balanced`). When sweeping multiple rates, you must rename this directory after each run, otherwise the next run overwrites it.

**The `&& mv` pattern in the tmux command handles this**, but it only works if the eval script exits successfully. If it crashes, the rename doesn't happen, and you must rename manually before the next run.

**Do NOT try to automate this with a sweep script** — the eval script's internal retry logic, server kill behavior, and directory management make sweep scripts extremely error-prone. Run one rate at a time.

### 6. AsyncMoE is much more stable than SGLang

In practice, AsyncMoE runs completed reliably at all rates with zero crashes. SGLang runs frequently crashed or produced partial results at higher rates. Plan accordingly — run AsyncMoE first to get clean data, then attempt SGLang.

## Analyzing Results

### Parser script

`experiments/evaluations/parse_detokenizer_logs.py` extracts throughput and ITL from server logs and computes **throughput-weighted aggregate metrics**. The weighting ensures steady-state periods (high throughput) dominate over ramp-up and drain periods.

```bash
conda activate disag12
python experiments/evaluations/parse_detokenizer_logs.py \
  experiments/sphere-16-tput-itl-mar30/ \
  --csv experiments/sphere-16-tput-itl-mar30/metrics.csv
```

### Log formats parsed

**AsyncMoE** (`server.log`):
```
Detokenizer: token throughput: 13.95k tokens/s | ITL mean=150.1ms p50=150.0ms p99=160.0ms
```

**SGLang** (`logs/server_head.log`):
```
[...] from Detokenizer Manager, Throughput: 12019.2 tokens/s, ..., ITL mean=147.34 ms, median=147.59 ms, p99=162.77 ms, samples=118680
```

### Output CSV columns

| Column | Description |
|---|---|
| `system` | `asyncmoe` or `sglang_ep16` |
| `workload` | `sharegpt_balanced` or `gsm8k_balanced` |
| `rate_rps` | request sending rate |
| `num_samples` | number of log lines (≈ seconds of data) |
| `peak_tput` | max instantaneous throughput (tokens/s) |
| `avg_tput` | simple average throughput |
| `weighted_itl_mean_ms` | throughput-weighted ITL mean |
| `weighted_itl_median_ms` | throughput-weighted ITL median |
| `weighted_itl_p99_ms` | throughput-weighted ITL p99 |

## Output Directory Layout

```
experiments/sphere-16-tput-itl-mar30/
├── checklist.txt                          # experiment plan with results
├── metrics.csv                            # aggregated metrics from parser
├── asyncmoe-gptoss-results/
│   ├── asyncmoe-sharegpt_balanced-25rps/
│   │   ├── server.log                     # ← parse this for metrics
│   │   ├── server_cmd.sh
│   │   ├── bench_cmd.sh
│   │   └── result.json
│   ├── asyncmoe-sharegpt_balanced-50rps/
│   │   └── ...
│   └── ...
├── sglang-gptoss-results/
│   ├── sglang_ep16-gsm8k_balanced-25rps/
│   │   ├── logs/
│   │   │   └── server_head.log            # ← parse this for metrics
│   │   ├── bench_result.json
│   │   ├── bench_cmd.sh
│   │   └── server_cmd.sh
│   └── ...
├── asyncmoe-*.log                         # per-run runner logs
└── sglang_ep16-*.log                      # per-run runner logs
```
