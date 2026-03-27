#!/usr/bin/bash
# config.sh — Fixed cluster / model / runtime / benchmark config for Sphere-16 EP16
# Source this file; do not execute directly.

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/DisagMoE"
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

HEAD_NODE="sgpu0"
HEAD_IP="10.0.0.1"
WORKER_NODES=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)

# ── Model — gptoss_120b full config (36 layers, 128 experts, top-4, bf16) ────
MODEL_NAME="gptoss_120b"
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"

# ── Runtime ───────────────────────────────────────────────────────────────────
TRANSPORT="zmq"
HOST_IFNAME="ens1f1np1"
NCCL_IB_HCA="mlx5_1"
NCCL_IB_GID_INDEX="3"
PLACEMENT="colocate"
DP_SIZE=$WORLD_SIZE
EP_SIZE=$WORLD_SIZE
MEM_FRAC=0.98
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=1024
MAX_PENDING_SENDS=16
BLOCK_SIZE=16

# ── Scheduler ─────────────────────────────────────────────────────────────────
UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=4
DEFRAG_LOOKBACK_STEPS=4

# ── Benchmark — 10 000 requests, 2000 rps, lengths 256-512 uniform ────────────
BENCH_RATE=2000
BENCH_TIME=5
BENCH_MIN_IN=256
BENCH_MAX_IN=512
BENCH_MIN_OUT=256
BENCH_MAX_OUT=512
BENCH_CURL_TIMEOUT=600

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=300   # 5 min — no NFS contention on Sphere
