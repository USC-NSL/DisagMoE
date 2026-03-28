#!/usr/bin/bash
# config.sh — Fixed cluster / model / runtime / benchmark config for Delta EP16
# Source this file; do not execute directly.
#
# Usage: source experiments/scripts/delta/eval/config.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LOG_BASE="$REPO_DIR/experiments/amoe-081"
# RESULTS_DIR is NOT set here — ep16_eval.sh requires it as $1.
GATING_DIR="$REPO_DIR/gating_profiles"
MINICONDA="/projects/bgro/spark36/miniconda3"
CONDA_ENV="amoe"
SERVER_PORT=6699

# ── System identity ───────────────────────────────────────────────────────────
SYSTEM_NAME="asyncmoe"    # used as the prefix in per-run directory names

# ── Cluster — 4-node × 4-GPU A100-SXM4-40GB (Delta gpuA100x4) ───────────────
N_NODE=4
N_GPU_PER_NODE=4
WORLD_SIZE=16

# ── Model — gptoss_120b full config (36 layers, 128 experts, top-4, bf16) ────
MODEL_NAME="gptoss_120b"
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"
# NUM_LAYERS not set — model default (36) is the full config

# ── Runtime ───────────────────────────────────────────────────────────────────
TRANSPORT="zmq"
HOST_IFNAME="hsn0"       # HPE Slingshot NIC for NCCL data plane
PLACEMENT="colocate"
DP_SIZE=$WORLD_SIZE
EP_SIZE=$WORLD_SIZE
MEM_FRAC=0.98            # Initial fraction; reduced on OOM retries
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=1024
MAX_PENDING_SENDS=16
BLOCK_SIZE=16

# ── Scheduler ─────────────────────────────────────────────────────────────────
UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=4
DEFRAG_LOOKBACK_STEPS=4

# ── Benchmark — 10 000 requests, 2000 rps, dataset generator (sharegpt) ───────
BENCH_RATE=${BENCH_RATE:-2000}
BENCH_TIME=${BENCH_TIME:-5}            # 2000 rps × 5 s = 10 000 requests
BENCH_GENERATOR=${BENCH_GENERATOR:-"dataset"}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-"$REPO_DIR/datasets/sharegpt_lengths.npy"}
BENCH_MAX_SEQ_LEN=${BENCH_MAX_SEQ_LEN:-2048}
BENCH_MIN_IN=${BENCH_MIN_IN:-256}
BENCH_MAX_IN=${BENCH_MAX_IN:-512}
BENCH_MIN_OUT=${BENCH_MIN_OUT:-256}
BENCH_MAX_OUT=${BENCH_MAX_OUT:-512}
BENCH_CURL_TIMEOUT=${BENCH_CURL_TIMEOUT:-600}   # 10-min hard cap; run expected to finish in <6 min

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=1200  # 20 min — NFS import contention on Delta can be slow
