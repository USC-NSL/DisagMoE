OUTPUT_LEN=16
N_NODE=1
N_GPU_PER_NODE=2
NUM_LAYERS=4
NUM_EXPERTS=4
MAX_BATCH_SIZE_ATTN=16
MAX_BATCH_SIZE_EXP=16
GRAPH_STRIDE=16
step_attn=1
dp_size=1
step_exp=1
ep_size=1
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
    --analyze-throughput