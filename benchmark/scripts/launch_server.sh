OUTPUT_LEN=300
N_NODE=2
N_GPU_PER_NODE=4
NUM_LAYERS=32
NUM_EXPERTS=8
MAX_BATCH_SIZE_ATTN=192
MAX_BATCH_SIZE_EXP=256
GRAPH_STRIDE=8
step_attn=1
dp_size=4
step_exp=1
ep_size=4
top_k=1

if [ ! -d "./reports" ]; then
    mkdir -p ./reports
fi

REPORT_DIR=./reports/server.csv

python benchmark/server.py \
    -o $OUTPUT_LEN \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -K $top_k \
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
    --analyze-throughput \
    --trace