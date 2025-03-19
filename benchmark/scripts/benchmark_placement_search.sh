#!/bin/bash

# profile & report configs
export REPORT_DIR=./reports/server.csv


# model configs
export NUM_LAYERS=32
export NUM_EXPERTS=8
export MAX_BATCH_SIZE_ATTN=256
export MAX_BATCH_SIZE_EXP=512
export GRAPH_STRIDE=16
export N_REQUESTS=6000
export OUTPUT_LEN=128

# parallel configs
export N_GPU_ATTN=8
export N_GPU_EXP=8

N_GPU_PER_NODE=4
N_NODE=$(((N_GPU_ATTN + N_GPU_EXP) / N_GPU_PER_NODE))

echo N_NODE: $N_NODE, N_GPU_PER_NODE: $N_GPU_PER_NODE

step_attns=($(for ((i=1; i<=N_GPU_ATTN; i*=2)); do echo $i; done))
dp_sizes=($(for i in "${step_attns[@]}"; do echo $((N_GPU_ATTN / i)); done))

echo step_attns: ${step_attns[@]}
echo dp_sizes: ${dp_sizes[@]}

step_exps=($(for ((i=1; i<=N_GPU_EXP; i*=2)); do echo $i; done))
ep_sizes=($(for i in "${step_exps[@]}"; do echo $((N_GPU_EXP / i)); done))

echo step_exps: ${step_exps[@]}
echo ep_sizes: ${ep_sizes[@]}

for step_attn in "${step_attns[@]}"; do
    for step_exp in "${step_exps[@]}"; do
        dp_size=$((N_GPU_ATTN / step_attn))
        ep_size=$((N_GPU_EXP / step_exp))
        echo "!!![bash script]!!!" running with step_attn: $step_attn, step_exp: $step_exp, dp_size: $dp_size, ep_size: $ep_size
        python benchmark/benchmark_serving.py \
            -o $OUTPUT_LEN \
            -n $N_REQUESTS \
            -N $N_NODE \
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
            --file $REPORT_DIR
    done
done