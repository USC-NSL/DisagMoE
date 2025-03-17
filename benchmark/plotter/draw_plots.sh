DP_SIZE=1
EP_SIZE=2
NUM_NODES=1

RATES=(10)

for rate in "${RATES[@]}"; do
    python benchmark/plotter/output_req.py \
        --dp-size $DP_SIZE \
        --ep-size $EP_SIZE \
        --num-nodes $NUM_NODES \
        --rate $rate
    python benchmark/plotter/sampler_step.py \
        --dp-size $DP_SIZE \
        --ep-size $EP_SIZE \
        --num-nodes $NUM_NODES \
        --rate $rate

    python benchmark/plotter/queueing_delay.py \
        --dp-size $DP_SIZE \
        --ep-size $EP_SIZE \
        --num-nodes $NUM_NODES \
        --rate $rate

    python benchmark/plotter/ttft.py \
        --dp-size $DP_SIZE \
        --ep-size $EP_SIZE \
        --num-nodes $NUM_NODES \
        --rate $rate
done