#!/bin/bash

# profile & report configs
REPORT_DIR=./reports/server.csv

# model configs
NUM_LAYERS=32
NUM_EXPERTS=8
MAX_BATCH_SIZE_ATTN=192
MAX_BATCH_SIZE_EXP=512
GRAPH_STRIDE=16
N_REQUESTS=500
OUTPUT_LEN=128
INPUT_LEN=1

# parallel configs
N_GPU_ATTN=8
N_GPU_EXP=8

N_GPU_PER_NODE=8
N_NODE=1

step_attn=2
dp_size=2
step_exp=1
ep_size=4
top_k=1

python benchmark/benchmark_serving.py \
    -i $INPUT_LEN \
    -o $OUTPUT_LEN \
    -n $N_REQUESTS \
    --rate 10 \
    -N $N_NODE \
    -K $top_k \
    -g $N_GPU_PER_NODE \
    --num-layers $NUM_LAYERS \
    --num-experts $NUM_EXPERTS \
    --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
    --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
    --graph-stride $GRAPH_STRIDE \
    --step-attn $step_attn \
    --step-exp $step_exp \
    --dp-size $dp_size \
    --ep-size $ep_size \
    -ca \
    --file $REPORT_DIR \
    --trace
