#!/usr/bin/bash
set -euo pipefail

N_NODE=3
N_GPU_PER_NODE=2
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))

MODEL_NAME="gptoss_120b"
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"

NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
top_k="${top_k:-2}"

MODEL_ARGS="--model $MODEL_NAME"
if [ -n "${NUM_LAYERS:-}" ]; then
    MODEL_ARGS="$MODEL_ARGS --num-layers $NUM_LAYERS"
fi
if [ -n "${NUM_EXPERTS:-}" ]; then
    MODEL_ARGS="$MODEL_ARGS --num-experts $NUM_EXPERTS"
fi
if [ -n "${NUM_KV_HEADS:-}" ]; then
    MODEL_ARGS="$MODEL_ARGS --num-kv-heads $NUM_KV_HEADS"
fi
if [ -n "${top_k:-}" ]; then
    MODEL_ARGS="$MODEL_ARGS --topk $top_k"
fi
if [ -n "${ATTN_QKV_QUANT:-}" ]; then
    MODEL_ARGS="$MODEL_ARGS --attn-qkv-quant $ATTN_QKV_QUANT"
fi
if [ -n "${MOE_LINEAR_QUANT:-}" ]; then
    MODEL_ARGS="$MODEL_ARGS --moe-linear-quant $MOE_LINEAR_QUANT"
fi

placement="colocate"

ENABLE_ASYMMETRIC_DEPLOYMENT=1
EXPERT_ALLOCATION_FILE="experiments/scripts/sphere-3/asym_expert_alloc.json"

if [ "$ENABLE_ASYMMETRIC_DEPLOYMENT" -eq 1 ]; then
    if [ ! -f "$EXPERT_ALLOCATION_FILE" ]; then
        echo "expert allocation file not found: $EXPERT_ALLOCATION_FILE"
        exit 1
    fi
    MODEL_ARGS="$MODEL_ARGS --expert-allocation-path $EXPERT_ALLOCATION_FILE"
fi

echo "model args: $MODEL_ARGS"

transport_backend=zmq

HOST_IFNAME="ens1f1np1"
NCCL_IB_HCA="mlx5_1"
NCCL_IB_GID_INDEX="3"
export NCCL_RUNTIME_CONNECT="${NCCL_RUNTIME_CONNECT:-0}"

NETWORK_ARGS=""
if [ -n "$HOST_IFNAME" ]; then
    NETWORK_ARGS="--host-ifname $HOST_IFNAME"
fi
if [ -n "$NCCL_IB_HCA" ]; then
    NETWORK_ARGS="$NETWORK_ARGS --nccl-ib-hca $NCCL_IB_HCA"
fi
if [ -n "$NCCL_IB_GID_INDEX" ]; then
    NETWORK_ARGS="$NETWORK_ARGS --nccl-ib-gid-index $NCCL_IB_GID_INDEX"
fi

dp_size=$WORLD_SIZE
ep_size=$WORLD_SIZE
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=512

UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=4
DEFRAG_LOOKBACK_STEPS=4

if [ "$placement" == "colocate" ]; then
    dp_size=$WORLD_SIZE
    ep_size=$WORLD_SIZE
fi

LESS_THAN_SM90=1
ENABLE_CUDA_GRAPH_ATTN=1
ENABLE_CUDA_GRAPH_EXPERT=1
ENABLE_TORCH_PROFILE=0

USE_SERIAL_GEMM_MOE=0

GATE_PROFILE_FILE=""

ENABLE_ADVANCED_LOGGING=1
ADVANCED_LOGGING_DIR="./advanced_logs"
ADVANCED_LOGGING_SAMPLE_RATE=0.1

REPORT_DIR=./reports

if [ ! -d "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR"
fi

CUDA_GRAPH_ATTN_ARGS=""
if [ "$ENABLE_CUDA_GRAPH_ATTN" -eq 1 ]; then
    CUDA_GRAPH_ATTN_ARGS="--cuda-graph-attn"
fi

CUDA_GRAPH_EXPERT_ARGS=""
if [ "$ENABLE_CUDA_GRAPH_EXPERT" -eq 1 ]; then
    CUDA_GRAPH_EXPERT_ARGS="--cuda-graph-expert"
fi

LESS_THAN_SM90_ARGS=""
if [ "$LESS_THAN_SM90" -eq 1 ]; then
    LESS_THAN_SM90_ARGS="--less-than-sm90"
fi

SERIAL_GEMM_ARGS=""
if [ "$USE_SERIAL_GEMM_MOE" -eq 1 ]; then
    SERIAL_GEMM_ARGS="--serial-gemm"
fi

ADVANCED_LOGGING_ARGS=""
if [ "$ENABLE_ADVANCED_LOGGING" -eq 1 ]; then
    ADVANCED_LOGGING_ARGS="--enable-advanced-logging --advanced-logging-dir $ADVANCED_LOGGING_DIR --advanced-logging-sample-rate $ADVANCED_LOGGING_SAMPLE_RATE"
fi

UNIFIED_SCHEDULER_ARGS=""
if [ "$placement" == "colocate" ]; then
    UNIFIED_SCHEDULER_ARGS="--unified-scheduler-type $UNIFIED_SCHEDULER_TYPE \
 --defrag-weight-decay $DEFRAG_WEIGHT_DECAY \
 --defrag-lookahead-steps $DEFRAG_LOOKAHEAD_STEPS \
 --defrag-lookback-steps $DEFRAG_LOOKBACK_STEPS"
fi

REPORT_TABLE=$REPORT_DIR/benchmark.csv

python benchmark/server.py \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -u 0.98 \
    $MODEL_ARGS \
    --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
    --max-attn-graph-bsz $MAX_BATCH_SIZE_ATTN \
    --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
    --block-size 16 \
    --placement $placement \
    --dp-size $dp_size \
    --ep-size $ep_size \
    --transport $transport_backend \
    $NETWORK_ARGS \
    $UNIFIED_SCHEDULER_ARGS \
    $SERIAL_GEMM_ARGS \
    $LESS_THAN_SM90_ARGS \
    $CUDA_GRAPH_ATTN_ARGS \
    $CUDA_GRAPH_EXPERT_ARGS \
    --file $REPORT_TABLE \
    --analyze-throughput \
    --trace \
    --gate-profile-file "$GATE_PROFILE_FILE" \
    $ADVANCED_LOGGING_ARGS
