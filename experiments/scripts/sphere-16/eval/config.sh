#!/usr/bin/bash
# config.sh — Shared cluster / runtime / benchmark config for Sphere-16 EP16
# Source this file from a model-specific config; do not execute directly.
#
# Model-specific variables (MODEL_NAME, quant settings, shared-expert flags)
# are set in gptoss_config.sh / glm45air_config.sh, which source this file.

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
GATING_DIR="$REPO_DIR/gating_profiles"
MINICONDA="$HOME/miniconda3"
CONDA_ENV="disag12"
SERVER_PORT=6699

# ── System identity ───────────────────────────────────────────────────────────
SYSTEM_NAME="asyncmoe"

# ── Cluster — 8 nodes × 2 L40S GPUs = EP16 ───────────────────────────────────
N_NODE=8
N_GPU_PER_NODE=2
WORLD_SIZE=16

HEAD_NODE="sgpu2"
HEAD_IP="10.0.0.1"
WORKER_NODES=(sgpu3 sgpu4 sgpu5 sgpu6 sgpu7 sgpu8 sgpu9)

# ── Runtime ───────────────────────────────────────────────────────────────────
TRANSPORT="zmq"
HOST_IFNAME="ens1f1np1"
NCCL_IB_HCA="mlx5_1"
NCCL_IB_GID_INDEX="3"
PLACEMENT="colocate"
DP_SIZE=$WORLD_SIZE
EP_SIZE=$WORLD_SIZE
MEM_FRAC=${MEM_FRAC:-0.98}
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=1024
MAX_PENDING_SENDS=16
BLOCK_SIZE=16

# ── Scheduler ─────────────────────────────────────────────────────────────────
UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=${DEFRAG_LOOKAHEAD_STEPS:-4}
DEFRAG_LOOKBACK_STEPS=${DEFRAG_LOOKBACK_STEPS:-4}

# ── Benchmark — 10 000 requests, 2000 rps, dataset generator (sharegpt) ───────
BENCH_RATE=${BENCH_RATE:-2000}
BENCH_TIME=${BENCH_TIME:-10}
BENCH_GENERATOR=${BENCH_GENERATOR:-"dataset"}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-"$REPO_DIR/datasets/sharegpt_lengths.npy"}
BENCH_MAX_CONTEXT_LEN=${BENCH_MAX_CONTEXT_LEN:-2048}
BENCH_MIN_IN=${BENCH_MIN_IN:-256}
BENCH_MAX_IN=${BENCH_MAX_IN:-512}
BENCH_MIN_OUT=${BENCH_MIN_OUT:-256}
BENCH_MAX_OUT=${BENCH_MAX_OUT:-512}
BENCH_CURL_TIMEOUT=${BENCH_CURL_TIMEOUT:-1200}

# ── Peak-state window for --analyze-throughput (seconds after benchmark start)
ANALYZE_THROUGHPUT_WINDOW="15,45"

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=300   # 5 min — no NFS contention on Sphere
