OUTPUT_LEN=1024
N_TIME=1
N_NODE=1
N_GPU_PER_NODE=4
NUM_LAYERS=16
NUM_EXPERTS=8
MAX_BATCH_SIZE_ATTN=128
MAX_BATCH_SIZE_EXP=512
GRAPH_STRIDE=32
step_attn=1
dp_size=4
step_exp=1
ep_size=4
placement=colocate
REPORT_DIR=/home/hogura1999/DisagMoE/reports/${placement}_poisson.csv

RATES=(15)

for rate in "${RATES[@]}"; do
    n_req=$((rate * N_TIME))
    echo "!!![bash script]!!!" running with rate: $rate
    python benchmark/benchmark_serving.py \
        -o $OUTPUT_LEN \
        -n $n_req \
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
        -c \
        --file $REPORT_DIR \
        --generator-type poisson \
        --rate $rate \
        --analyze-throughput \
        --placement $placement
done