#!/usr/bin/bash

# cluster config
N_NODE=1
N_GPU_PER_NODE=2
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))

# model config
NUM_LAYERS=16
NUM_EXPERTS=4
MODEL_NAME="mixtral"  # options: mixtral | qwen3_235b
top_k=1
ATTN_QKV_QUANT="none" # options: none | fp8
MOE_LINEAR_QUANT="none" # options: none | fp8

# placement config
placement="colocate"

# runtime config
transport_backend=zmq

dp_size=$WORLD_SIZE
ep_size=$WORLD_SIZE
MAX_BATCH_SIZE_ATTN=160
MAX_BATCH_SIZE_EXP=512

if [ $placement == "colocate" ]; then
    dp_size=$WORLD_SIZE
    ep_size=$WORLD_SIZE
fi

ENABLE_CUDA_GRAPH_ATTN=0

ENABLE_TORCH_PROFILE=0

USE_SERIAL_GEMM_MOE=0

# Optional: path to a gate profile file on the launching node. If set, it will be
# uploaded to the cluster and delivered via Ray's object store.
# When provided, the attention workers will use profile-driven gating.
# GATE_PROFILE_FILE="./gating_profiles/gating_sharegptv3_155.parquet"

# transport backend: zmq | ucx

REPORT_DIR=./reports
# Set to 1 to enable PyTorch profiler; 0 to disable
PROFILE_DIR=$REPORT_DIR/torch_profile

if [ ! -d $REPORT_DIR ]; then
    mkdir -p $REPORT_DIR
fi

# Conditionally enable profiler

CUDA_GRAPH_ATTN_ARGS=""
if [ "$ENABLE_CUDA_GRAPH_ATTN" -eq 1 ]; then
    CUDA_GRAPH_ATTN_ARGS="--cuda-graph-attn"
fi

PROFILE_ARGS=""
if [ "$ENABLE_TORCH_PROFILE" -eq 1 ]; then
    if [ ! -d $PROFILE_DIR ]; then
        mkdir -p $PROFILE_DIR
    fi
    PROFILE_ARGS="-p $PROFILE_DIR"
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
    -K $top_k \
    -u 0.75 \
    --num-kv-heads 4 \
    --num-layers $NUM_LAYERS \
    --num-experts $NUM_EXPERTS \
    --model $MODEL_NAME \
    --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
    --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
    --block-size 16 \
    --placement $placement \
    --dp-size $dp_size \
    --ep-size $ep_size \
    --transport $transport_backend \
    --attn-qkv-quant $ATTN_QKV_QUANT \
    --moe-linear-quant $MOE_LINEAR_QUANT \
    $SERIAL_GEMM_ARGS \
    $CUDA_GRAPH_ATTN_ARGS \
    --file $REPORT_TABLE \
    --analyze-throughput \
    --trace \
    --gate-profile-file "$GATE_PROFILE_FILE"
