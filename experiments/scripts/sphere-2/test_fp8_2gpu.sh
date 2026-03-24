#!/usr/bin/bash
# Test run: 1 node, 2 GPUs, 8 layers, FP8 attention + FP8 experts (CUTLASS SM89)
# Validated on sgpu5 (NVIDIA L40S) with conda env disag12

N_NODE=1
N_GPU_PER_NODE=2
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))

python benchmark/server.py \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -u 0.98 \
    --model gptoss_120b \
    --num-layers 8 \
    --attn-qkv-quant fp8 \
    --moe-linear-quant fp8 \
    --max-batch-size-attn 64 \
    --max-attn-graph-bsz 64 \
    --max-batch-size-expert 128 \
    --max-pending-sends 4 \
    --block-size 16 \
    --placement colocate \
    --dp-size $WORLD_SIZE \
    --ep-size $WORLD_SIZE \
    --transport zmq \
    --less-than-sm90 \
    --cuda-graph-attn \
    --cuda-graph-expert \
    --unified-scheduler-type defrag \
    --defrag-weight-decay 0.8 \
    --defrag-lookahead-steps 4 \
    --defrag-lookback-steps 4 \
    --analyze-throughput \
    --trace
