#!/usr/bin/bash
# Delta: 1-node, 4-GPU local test with reduced gptoss_120b layers
# Usage: source env.sh first, then bash this script

set -euo pipefail

# cluster config
N_NODE=1
N_GPU_PER_NODE=4
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))

# model config — reduced layers for quick local test
MODEL_NAME="gptoss_120b"
NUM_LAYERS=${NUM_LAYERS:-4}
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"

MODEL_ARGS="--model $MODEL_NAME --num-layers $NUM_LAYERS"
MODEL_ARGS="$MODEL_ARGS --attn-qkv-quant $ATTN_QKV_QUANT"
MODEL_ARGS="$MODEL_ARGS --moe-linear-quant $MOE_LINEAR_QUANT"

echo "model args: $MODEL_ARGS"

# placement config
placement="colocate"

# runtime config
transport_backend=zmq

# No network args needed for single-node (NVLink/PCIe handles intra-node)
NETWORK_ARGS=""

dp_size=$WORLD_SIZE
ep_size=$WORLD_SIZE
MAX_BATCH_SIZE_ATTN=32
MAX_BATCH_SIZE_EXP=128
MAX_PENDING_SENDS=16

UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=4
DEFRAG_LOOKBACK_STEPS=4

LESS_THAN_SM90=1
ENABLE_CUDA_GRAPH_ATTN=1
ENABLE_CUDA_GRAPH_EXPERT=1

REPORT_DIR=./reports
mkdir -p $REPORT_DIR

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

REPORT_TABLE=$REPORT_DIR/benchmark_local.csv

python benchmark/server.py \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -u 0.70 \
    $MODEL_ARGS \
    --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
    --max-attn-graph-bsz $MAX_BATCH_SIZE_ATTN \
    --max-pending-sends $MAX_PENDING_SENDS \
    --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
    --block-size 16 \
    --placement $placement \
    --dp-size $dp_size \
    --ep-size $ep_size \
    --transport $transport_backend \
    $NETWORK_ARGS \
    --unified-scheduler-type $UNIFIED_SCHEDULER_TYPE \
    --defrag-weight-decay $DEFRAG_WEIGHT_DECAY \
    --defrag-lookahead-steps $DEFRAG_LOOKAHEAD_STEPS \
    --defrag-lookback-steps $DEFRAG_LOOKBACK_STEPS \
    $LESS_THAN_SM90_ARGS \
    $CUDA_GRAPH_ATTN_ARGS \
    $CUDA_GRAPH_EXPERT_ARGS \
    --file $REPORT_TABLE \
    --analyze-throughput \
    --trace
