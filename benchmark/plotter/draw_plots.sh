DP_SIZE=2
EP_SIZE=4
NUM_NODES=1

RATES=(15 25)

for rate in "${RATES[@]}"; do
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