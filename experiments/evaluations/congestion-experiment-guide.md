# Network Congestion Tolerance Experiment Guide

## What This Experiment Does

Measures how AsyncMoE and SGLang EP16 perform under trace-driven network interference. The interference generator (`interference_gen/`) replays real cloud noise traces as RDMA or TCP traffic that competes with the serving system's network, while a benchmark drives requests at a controlled rate. Compare throughput and ITL with vs without interference.

## Prerequisites

### Cluster

- 8 nodes × 2 L40S = 16 GPUs (Sphere-16)
- RoCE via `ens1f1np1` (`mlx5_1` HCA), 200 Gbps per link
- No NFS — sync code via rsync

### Software

| What | AsyncMoE | SGLang |
|------|----------|--------|
| Conda env | `disag12` | `sglang-fp` |
| Repo | `~/DisagMoE` | `~/sglang-fake-prefill` |

### After every cluster reboot

```bash
for n in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh $n "cd ~/gdrcopy && sudo bash ./insmod.sh && sudo ldconfig"
done
```

Verify: `python -c "import disagmoe_c"` in `disag12` env, `python -c "import sglang"` in `sglang-fp` env on all nodes. sgpu7 may need full setup (miniconda, apt packages, `make pip`) if it was reimaged.

### perftest package

Calibration uses `ib_write_bw`. Install if missing on any node:
```bash
ssh <node> "sudo apt-get install -y perftest"
```

## Interference Generator

See `interference_gen/README.md` for full details. Key usage patterns below.

### Single-link (RDMA) — RECOMMENDED

Interferes on one specific link. Uses `UCX_TLS=rc` (RDMA), competes directly with NCCL traffic.

**Run from the LOCAL node** of the pair, not from sgpu0:

```bash
ssh sgpu4 'cd ~/DisagMoE/interference_gen && \
  ./run_interfere.sh --peer-host sgpu6 --peer-ip 10.0.0.5 \
    --trace aws_hpc_metal --link-capacity-gbps 200'
```

Stop:
```bash
ssh sgpu4 'cd ~/DisagMoE/interference_gen && \
  ./run_interfere.sh --peer-host sgpu6 --stop'
```

### Ring (all links) — TCP ONLY

> **⚠️ RDMA ring mode is BROKEN.** `UCX_TLS=rc` ring fails on 8 nodes — most connections get "Endpoint is not connected" / "Destination is unreachable" errors, some nodes show zero traffic. Always use `--transport tcp` for ring mode.

```bash
cd ~/DisagMoE/interference_gen
./run_ring.sh \
  --nodes sgpu0:10.0.0.1,sgpu2:10.0.0.2,sgpu3:10.0.0.3,sgpu4:10.0.0.4,sgpu6:10.0.0.5,sgpu7:10.0.0.6,sgpu8:10.0.0.7,sgpu9:10.0.0.8 \
  --trace aws_hpc_metal --link-capacity-gbps 200 \
  --transport tcp
```

Note: TCP interference gets lower NIC priority than RDMA via PFC/ECN, so it competes less directly with NCCL.

### Transport summary

| Mode | Use | Why |
|------|-----|-----|
| Single-link `rc` (RDMA) | ✅ Works | Competes directly with NCCL RDMA |
| Ring `tcp` | ✅ Works | All links, but TCP gets lower priority |
| Ring `rc` (RDMA) | ❌ Broken | Connection failures across 8 nodes |

### Calibration

Both modes auto-calibrate at startup (~2 min). **Do not launch the benchmark until calibration is done.**

```bash
grep "Calibration complete" <interference_log>
```

### Verifying interference is active

**Always verify before starting the benchmark.** Invalid interference = invalid results.

```bash
# Single-link:
grep "monitor" <log>
# Expect: "ib_write_bw: XX Gbps (baseline=YY, interference=ZZ Gbps)"

# Ring:
grep "Gbps" <log> | tail -10
# Expect non-zero TX/RX on every node
# "WARN (no traffic) — TX=0 Gbps" means that node has NO interference
```

### Available traces

| Name | Character | CoV |
|------|-----------|-----|
| `aws_hpc_metal` | Persistent moderate jitter | 0.039 |
| `azure_hpc_200g` | Rare mild dips | 0.0077 |
| `oracle_hpc` | Rare severe clustered bursts | 0.0076 |
| `deep_est_ib` | Near-zero control | 0.0009 |

## Running AsyncMoE

```bash
RATE=85; TIME=100; TAG="aws-single-link"; ROUND="round2"
OUTDIR=~/DisagMoE/experiments/<exp-dir>/asyncmoe-gptoss-results

# 1. Clean default dir
rm -rf $OUTDIR/asyncmoe-sharegpt_balanced

# 2. Launch in tmux (rename chained with &&)
tmux new-session -d -s amoe-run
tmux send-keys -t amoe-run "source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate disag12 && cd ~/DisagMoE && \
  BENCH_RATE=$RATE BENCH_TIME=$TIME BENCH_CURL_TIMEOUT=3600 \
  ANALYZE_THROUGHPUT_WINDOW=15,95 \
  bash experiments/scripts/sphere-16/eval/gptoss_eval.sh $OUTDIR \
    --only sharegpt_balanced \
  2>&1 | tee <exp-dir>/run-amoe-${RATE}rps-${TAG}-${ROUND}.log && \
  mv $OUTDIR/asyncmoe-sharegpt_balanced \
     $OUTDIR/asyncmoe-sharegpt_balanced-${RATE}rps-${TAG}-${ROUND} && \
  echo 'RENAME DONE'" Enter
```

For gsm8k: add `BENCH_DATASET_PATH=$HOME/DisagMoE/datasets/gsm8k_lengths.npy` and `--only gsm8k_balanced`.

**Health check**: detokenizer logs throughput every ~10s. No output for >60s = hung → kill and retry.

## Running SGLang

### Pre-flight (MANDATORY before EVERY run)

```bash
pkill -9 -f sglang
for n in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh $n "pkill -9 -f sglang; fuser -k 25000/tcp 25001/tcp 25002/tcp \
    25003/tcp 25004/tcp 25005/tcp 30000/tcp" &
done; wait; sleep 8
fuser 25000/tcp 30000/tcp  # must produce no output
```

### Single client (rates ≤100 rps)

```bash
RATE=100; PROMPTS=$((RATE * 100)); TAG="no-interference"; ROUND="round2"
OUTDIR=~/DisagMoE/experiments/<exp-dir>/sglang-gptoss-results

rm -rf $OUTDIR/sglang_ep16-sharegpt_balanced

tmux new-session -d -s sgrun
tmux send-keys -t sgrun "source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate sglang-fp && cd ~/sglang-fake-prefill && \
  BENCH_REQUEST_RATE=$RATE BENCH_NUM_PROMPTS=$PROMPTS \
  BENCH_TIMEOUT=3600 BENCH_DISABLE_STREAM=1 MAX_RETRIES=2 \
  bash experiments/sphere/eval/gptoss_eval.sh $OUTDIR \
    --only '_ep16-sharegpt_balanced' \
  2>&1 | tee <exp-dir>/run-sg-${RATE}rps-${TAG}-${ROUND}.log && \
  mv $OUTDIR/sglang_ep16-sharegpt_balanced \
     $OUTDIR/sglang_ep16-sharegpt_balanced-${RATE}rps-${TAG}-${ROUND} && \
  echo 'RENAME DONE'" Enter
```

### Distributed 4-client (rates ≥200 rps)

A single client cannot drive SGLang to full throughput above ~200 rps. Use the distributed driver:

```bash
RATE=2000; PROMPTS=20000; TAG="no-interference"; ROUND="round2"
bash ~/DisagMoE/experiments/<exp-dir>/run_sglang_dist_once.sh \
  ~/DisagMoE/experiments/<exp-dir>/sglang-gptoss-results/sglang_ep16-sharegpt_balanced-${RATE}rps-${TAG}-${ROUND} \
  ~/sglang-fake-prefill/gating_profiles/gptosss_balanced_output/balanced_gptoss120b_sharegpt_200.parquet \
  ~/sglang-fake-prefill/datasets/sharegpt_lengths.npy \
  $PROMPTS $RATE
```

This fans out to 4 hosts (sgpu0/2/3/4), each sending prompts/4 at rate/4, and aggregates results.

### SGLang-specific rules

- **`--disable-stream` is mandatory** — without it, HTTP connection exhaustion at sustained rates causes `ConnectionResetError`. ITL from `bench_result.json` will be zero; use `logs/server_head.log` for ITL data.
- **Ports 25000–25005 and 30000 must be freed before every run** — zombie processes hold them after kills.
- **Gloo crashes are transient** — `RuntimeError: Connection closed by peer` in `all_gather_into_tensor`. Just kill and retry. A cluster reboot can help if persistent.

## Per-Run Workflow

```
1. PRE-FLIGHT: kill processes, free ports, clean default output dirs
2. START INTERFERENCE (if needed): launch in tmux, wait for calibration, verify active
3. LAUNCH BENCHMARK: AsyncMoE eval script or SGLang single/distributed
4. MONITOR: 1 min for crash check, then every 5 min
5. AFTER COMPLETION: stop interference, verify results exist, rename output dir
6. UPDATE METRICS: run parse_detokenizer_logs.py (see below)
```

## Analyzing Results

### Parser

```bash
conda activate disag12 && cd ~/DisagMoE

# Full metrics
python experiments/evaluations/parse_detokenizer_logs.py \
  experiments/<exp-dir>/ --csv experiments/<exp-dir>/metrics.csv

# Steady-state window (e.g., 15s–60s)
python experiments/evaluations/parse_detokenizer_logs.py \
  experiments/<exp-dir>/ --window 15,60 --csv experiments/<exp-dir>/metrics_15_60s.csv
```

### What it parses

**AsyncMoE** (`server.log`): `token throughput: 13.95k tokens/s | ITL mean=150.1ms p50=150.0ms p99=160.0ms`

**SGLang** (`logs/server_head.log`): `Throughput: 12019.2 tokens/s, ..., ITL mean=147.34 ms, median=147.59 ms, p99=162.77 ms`

### Output CSV columns

`system, workload, rate_rps, tag, num_samples, peak_tput, avg_tput, weighted_itl_mean_ms, weighted_itl_median_ms, weighted_itl_p99_ms`

### Directory naming

```
<system>-<workload>-<rate>rps-<tag>[-<round>]
```
The parser regex matches this automatically. The `tag` captures the interference condition and round.

## Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Interference log shows zero traffic on some nodes | RDMA ring broken | Use `--transport tcp` for ring |
| SGLang "port already in use" | Zombie processes | `fuser -k 25000-25005,30000/tcp` on all nodes |
| SGLang Gloo crash on startup | Transient hardware flakiness | Kill and retry; reboot cluster if persistent |
| `ib_write_bw: command not found` | `perftest` not installed | `sudo apt-get install -y perftest` |
| `libzmq.so.5 not found` after reboot | ldconfig wiped | `sudo ldconfig` on all nodes |
| `gdrdrv` not loaded after reboot | Kernel module wiped | `cd ~/gdrcopy && sudo bash ./insmod.sh` |
| Eval script overwrites previous results | Fixed output dir name | Always chain `&& mv` in tmux command |
| AsyncMoE hangs silently | System deadlock | No detokenizer output for >60s → kill and retry |
| Single client can't drive SGLang >200 rps | HTTP bottleneck | Use `run_sglang_dist_once.sh` with 4 clients |
