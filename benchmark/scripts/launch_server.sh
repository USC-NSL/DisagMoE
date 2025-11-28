#!/usr/bin/bash

# cluster config
N_NODE=1
N_GPU_PER_NODE=4
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))

# model config

MODEL_NAME="qwen3_235b"  # options: mixtral | qwen3_235b
ATTN_QKV_QUANT="fp8" # options: none | fp8
MOE_LINEAR_QUANT="fp8" # options: none | fp8

MODEL_ARGS="--model $MODEL_NAME"
if [ ! -z $NUM_LAYERS ]; then
    MODEL_ARGS="$MODEL_ARGS --num-layers $NUM_LAYERS"
fi
if [ ! -z $NUM_EXPERTS ]; then
    MODEL_ARGS="$MODEL_ARGS --num-experts $NUM_EXPERTS"
fi
if [ ! -z $NUM_KV_HEADS ]; then
    MODEL_ARGS="$MODEL_ARGS --num-kv-heads $NUM_KV_HEADS"
fi
if [ ! -z $top_k ]; then
    MODEL_ARGS="$MODEL_ARGS --topk $top_k"
fi
if [ ! -z $ATTN_QKV_QUANT ]; then
    MODEL_ARGS="$MODEL_ARGS --attn-qkv-quant $ATTN_QKV_QUANT"
fi
if [ ! -z $MOE_LINEAR_QUANT ]; then
    MODEL_ARGS="$MODEL_ARGS --moe-linear-quant $MOE_LINEAR_QUANT"
fi

echo "model args: $MODEL_ARGS"

# placement config
placement="colocate"

# runtime config
transport_backend=zmq

dp_size=$WORLD_SIZE
ep_size=$WORLD_SIZE
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=512

if [ $placement == "colocate" ]; then
    dp_size=$WORLD_SIZE
    ep_size=$WORLD_SIZE
fi

ENABLE_CUDA_GRAPH_ATTN=1

ENABLE_TORCH_PROFILE=0

USE_SERIAL_GEMM_MOE=0

# Optional: path to a gate profile file on the launching node. If set, it will be
# uploaded to the cluster and delivered via Ray's object store.
# When provided, the attention workers will use profile-driven gating.
# GATE_PROFILE_FILE="./gating_profiles/gating_sharegptv3_155.parquet"

# transport backend: zmq | ucx

REPORT_DIR=./reports

if [ ! -d $REPORT_DIR ]; then
    mkdir -p $REPORT_DIR
fi

# Conditionally enable profiler

CUDA_GRAPH_ATTN_ARGS=""
if [ "$ENABLE_CUDA_GRAPH_ATTN" -eq 1 ]; then
    CUDA_GRAPH_ATTN_ARGS="--cuda-graph-attn"
fi

SERIAL_GEMM_ARGS=""
if [ "$USE_SERIAL_GEMM_MOE" -eq 1 ]; then
    SERIAL_GEMM_ARGS="--serial-gemm"
fi

REPORT_TABLE=$REPORT_DIR/benchmark.csv

python benchmark/server.py \
    $PROFILE_ARGS \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -u 0.7 \
    $MODEL_ARGS \
    --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
    --max-attn-graph-bsz $MAX_BATCH_SIZE_ATTN \
    --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
    --block-size 16 \
    --placement $placement \
    --dp-size $dp_size \
    --ep-size $ep_size \
    --transport $transport_backend \
    $SERIAL_GEMM_ARGS \
    $CUDA_GRAPH_ATTN_ARGS \
    --file $REPORT_TABLE \
    --analyze-throughput \
    --trace \
    --gate-profile-file "$GATE_PROFILE_FILE"
