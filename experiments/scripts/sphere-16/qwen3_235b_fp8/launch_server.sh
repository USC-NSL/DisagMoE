#!/usr/bin/bash

# ──────────────────────────────────────────────────────────────
# Sphere-16: Qwen3-235B FP8 Server Launch
# Experiment: amoe-080
# Cluster:  8 nodes × 2 L40S GPUs = 16 GPUs
# Model:    qwen3_235b with FP8 quantization (attn + MoE)
# ──────────────────────────────────────────────────────────────

# cluster config
N_NODE=8
N_GPU_PER_NODE=2
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))

# model config
MODEL_NAME="qwen3_235b"
ATTN_QKV_QUANT="fp8"
MOE_LINEAR_QUANT="fp8"

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

# placement config
placement="colocate"

# Asymmetric Deployment Macro
ENABLE_ASYMMETRIC_DEPLOYMENT=0
EXPERT_ALLOCATION_FILE="benchmark/scripts/asym_alloc_config.json"

if [ "$ENABLE_ASYMMETRIC_DEPLOYMENT" -eq 1 ]; then
    if [ ! -f "$EXPERT_ALLOCATION_FILE" ]; then
        echo "expert allocation file not found: $EXPERT_ALLOCATION_FILE"
        exit 1
    fi
    MODEL_ARGS="$MODEL_ARGS --expert-allocation-path $EXPERT_ALLOCATION_FILE"
fi

echo "model args: $MODEL_ARGS"

# runtime config
transport_backend=zmq

HOST_IFNAME="ens1f1np1"  # network interface for inter-node IP and NCCL sockets
NCCL_IB_HCA="mlx5_1"    # IB/RoCE HCA device for NCCL data transfers
NCCL_IB_GID_INDEX="3"   # RoCE GID index matching the data network subnet
export NCCL_RUNTIME_CONNECT="${NCCL_RUNTIME_CONNECT:-0}"

NETWORK_ARGS=""
if [ ! -z "$HOST_IFNAME" ]; then
    NETWORK_ARGS="--host-ifname $HOST_IFNAME"
fi
if [ ! -z "$NCCL_IB_HCA" ]; then
    NETWORK_ARGS="$NETWORK_ARGS --nccl-ib-hca $NCCL_IB_HCA"
fi
if [ ! -z "$NCCL_IB_GID_INDEX" ]; then
    NETWORK_ARGS="$NETWORK_ARGS --nccl-ib-gid-index $NCCL_IB_GID_INDEX"
fi

dp_size=$WORLD_SIZE
ep_size=$WORLD_SIZE
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=1024
MAX_PENDING_SENDS=16

# UNIFIED_SCHEDULER_TYPE: flfs | defrag; only valid for colocate mode
UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=4
DEFRAG_LOOKBACK_STEPS=4

if [ $placement == "colocate" ]; then
    dp_size=$WORLD_SIZE
    ep_size=$WORLD_SIZE
fi

LESS_THAN_SM90=1 # L40S is SM89 (Ada Lovelace), disable deep_gemm
ENABLE_CUDA_GRAPH_ATTN=1
ENABLE_CUDA_GRAPH_EXPERT=1
ENABLE_TORCH_PROFILE=0

USE_SERIAL_GEMM_MOE=0

# Gate profile for qwen3_235b (sharegpt workload)
GATE_PROFILE_FILE="./gating_profiles/qwen3_235b_profiles/gating_qwen3_235b_sharegpt_200.parquet"

ENABLE_ADVANCED_LOGGING=0
ADVANCED_LOGGING_DIR="./advanced_logs"
ADVANCED_LOGGING_SAMPLE_RATE=0.1

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
    $PROFILE_ARGS \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -u 0.95 \
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
