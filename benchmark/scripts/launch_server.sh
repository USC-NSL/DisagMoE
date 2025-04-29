MIN_INPUT_LEN=100
MAX_INPUT_LEN=300
MIN_OUTPUT_LEN=100
MAX_OUTPUT_LEN=500
N_NODE=2
N_GPU_PER_NODE=4
NUM_LAYERS=32
NUM_EXPERTS=8
MAX_BATCH_SIZE_ATTN=160
MAX_BATCH_SIZE_EXP=512
GRAPH_STRIDE=8
step_attn=1
dp_size=4
step_exp=1
ep_size=4
top_k=1
placement=pipeline

REPORT_DIR=./reports

if [ ! -d $REPORT_DIR ]; then
    mkdir -p $REPORT_DIR
fi

REPORT_TABLE=$REPORT_DIR/benchmark.csv

python benchmark/server.py \
    --min-input-len $MIN_INPUT_LEN \
    --max-input-len $MAX_INPUT_LEN \
    --min-output-len $MIN_OUTPUT_LEN \
    --max-output-len $MAX_OUTPUT_LEN \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -K $top_k \
    -u 0.65 \
    --num-layers $NUM_LAYERS \
    --num-experts $NUM_EXPERTS \
    --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
    --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
    --graph-stride $GRAPH_STRIDE \
    --step-attn $step_attn \
    --step-exp $step_exp \
    --dp-size $dp_size \
    --ep-size $ep_size \
    --file $REPORT_TABLE \
    --analyze-throughput \
    --placement $placement \
    --trace