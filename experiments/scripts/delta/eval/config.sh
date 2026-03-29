#!/usr/bin/bash
# config.sh — Shared cluster / runtime / benchmark config for Delta EP16
# Source this file from a model-specific config; do not execute directly.
#
# Model-specific variables (MODEL_NAME, quant settings, shared-expert flags)
# are set in gptoss_config.sh / glm45air_config.sh, which source this file.

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# RESULTS_DIR is NOT set here — eval scripts require it as $1.
GATING_DIR="$REPO_DIR/gating_profiles"
MINICONDA="$HOME/miniconda3"
CONDA_ENV="amoe"
SERVER_PORT=6699

# ── System identity ───────────────────────────────────────────────────────────
SYSTEM_NAME="asyncmoe"    # used as the prefix in per-run directory names

# ── Cluster — 4-node × 4-GPU A100-SXM4-40GB (Delta gpuA100x4) ───────────────
N_NODE=4
N_GPU_PER_NODE=4
WORLD_SIZE=16

# ── Runtime ───────────────────────────────────────────────────────────────────
TRANSPORT="zmq"
HOST_IFNAME="hsn0"       # HPE Slingshot NIC for NCCL data plane
PLACEMENT="colocate"
DP_SIZE=$WORLD_SIZE
EP_SIZE=$WORLD_SIZE
MEM_FRAC=0.92            # Initial fraction; reduced on OOM retries
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=1024
MAX_PENDING_SENDS=16
BLOCK_SIZE=16

# ── Scheduler ─────────────────────────────────────────────────────────────────
UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=4
DEFRAG_LOOKBACK_STEPS=4

# ── Benchmark — 10 000 requests, 2000 rps, dataset generator (auto-selected per experiment) ──
BENCH_RATE=${BENCH_RATE:-2000}
BENCH_TIME=${BENCH_TIME:-5}            # 2000 rps × 5 s = 10 000 requests
BENCH_GENERATOR=${BENCH_GENERATOR:-"dataset"}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-"$REPO_DIR/datasets/sharegpt_lengths.npy"}
BENCH_MAX_CONTEXT_LEN=${BENCH_MAX_CONTEXT_LEN:-2048}
BENCH_MIN_IN=${BENCH_MIN_IN:-256}
BENCH_MAX_IN=${BENCH_MAX_IN:-512}
BENCH_MIN_OUT=${BENCH_MIN_OUT:-256}
BENCH_MAX_OUT=${BENCH_MAX_OUT:-512}
BENCH_CURL_TIMEOUT_SHAREGPT=${BENCH_CURL_TIMEOUT_SHAREGPT:-600}
BENCH_CURL_TIMEOUT_GSM8K=${BENCH_CURL_TIMEOUT_GSM8K:-300}

# ── Peak-state window for --analyze-throughput (seconds after benchmark start)
ANALYZE_THROUGHPUT_WINDOW="15,60"

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=600
