MIN_INPUT_LEN=10
MAX_INPUT_LEN=11
MIN_OUTPUT_LEN=50
MAX_OUTPUT_LEN=51
N_NODE=1
N_GPU_PER_NODE=2
NUM_LAYERS=16
NUM_EXPERTS=4
MAX_BATCH_SIZE_ATTN=160
MAX_BATCH_SIZE_EXP=512
GRAPH_STRIDE=8
step_attn=1
step_exp=1
dp_size=1
ep_size=1
top_k=1

transport_backend=ucx

placement="colocate"

if [ $placement == "colocate" ]; then
    dp_size=$((N_GPU_PER_NODE * N_NODE))
    ep_size=$dp_size
fi

# Optional: path to a gate profile file on the launching node. If set, it will be
# uploaded to the cluster and delivered via Ray's object store.
# When provided, the attention workers will use profile-driven gating.
GATE_PROFILE_FILE=""

# transport backend: zmq | ucx

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
    -u 0.75 \
    --num-kv-heads 4 \
    --num-layers $NUM_LAYERS \
    --num-experts $NUM_EXPERTS \
    --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
    --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
    --graph-stride $GRAPH_STRIDE \
    --block-size 16 \
    --step-attn $step_attn \
    --step-exp $step_exp \
    --placement $placement \
    --dp-size $dp_size \
    --ep-size $ep_size \
    --transport $transport_backend \
    --file $REPORT_TABLE \
    --analyze-throughput \
    --trace \
    --gate-profile-file "$GATE_PROFILE_FILE"
