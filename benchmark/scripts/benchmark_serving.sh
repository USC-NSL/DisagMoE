#!/bin/bash

MIN_INPUT_LEN=1
MAX_INPUT_LEN=10
MIN_OUTPUT_LEN=50
MAX_OUTPUT_LEN=51
N_NODE=1
N_GPU_PER_NODE=4
NUM_LAYERS=32
NUM_EXPERTS=4
MAX_BATCH_SIZE_ATTN=160
MAX_BATCH_SIZE_EXP=512
GRAPH_STRIDE=8
step_attn=1
dp_size=2
step_exp=1
ep_size=2
top_k=1

N_REQUESTS=1

# profile & report configs
REPORT_DIR=./reports/
if [ ! -d $REPORT_DIR ]; then
    mkdir -p $REPORT_DIR
fi

REPORT_FILE=$REPORT_DIR/benchmark_server.csv

python benchmark/benchmark_serving.py \
    --min-input-len $MIN_INPUT_LEN \
    --max-input-len $MAX_INPUT_LEN \
    --min-output-len $MIN_OUTPUT_LEN \
    --max-output-len $MAX_OUTPUT_LEN \
    -n $N_REQUESTS \
    --rate 10 \
    -N $N_NODE \
    -K $top_k \
    -u 0.65 \
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
    --file $REPORT_FILE \
    --analyze-throughput \
    --trace