#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/yizhuoliang/miniconda3/envs/disag12/bin/python"
BENCH="$SCRIPT_DIR/bench_fanin.py"

HOSTS=(sgpu6 sgpu1 sgpu3 sgpu4 sgpu5 sgpu7 sgpu8 sgpu9)
ALL_IPS="10.0.0.5,10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.6,10.0.0.7,10.0.0.8"
RECEIVER_IP="10.0.0.5"
IFNAME="ens1f1np1"
WORLD_SIZE=8

ITERS=500
WARMUP=20
PIPELINE_DEPTH=16
OUTDIR="$SCRIPT_DIR/results_fanin_8node"
mkdir -p "$OUTDIR"

MSG_SIZES=(65536 131072 262144 524288)
BACKENDS=(nccl nccl_gather uccl nixl)

for backend in "${BACKENDS[@]}"; do
    for sz in "${MSG_SIZES[@]}"; do
        echo "=== $backend / $(numfmt --to=iec $sz) / fan-in 7->1 ==="

        MASTER_PORT=$((31000 + RANDOM % 1000))
        NIXL_PORT=$((16000 + RANDOM % 1000))
        PIDS=()

        for rank in $(seq 0 $((WORLD_SIZE - 1))); do
            host="${HOSTS[$rank]}"
            OUT="$OUTDIR/${backend}_${sz}B_rank${rank}.json"

            if [ "$rank" -eq 0 ]; then
                $PYTHON "$BENCH" \
                    --rank $rank --world-size $WORLD_SIZE --backend "$backend" --msg-bytes "$sz" \
                    --iters "$ITERS" --warmup "$WARMUP" --pipeline-depth "$PIPELINE_DEPTH" \
                    --master-addr "$RECEIVER_IP" --master-port "$MASTER_PORT" \
                    --ifname "$IFNAME" --all-ips "$ALL_IPS" --nixl-port "$NIXL_PORT" \
                    --out "$OUT" &
                PIDS+=($!)
            else
                ssh "$host" "PATH=/home/yizhuoliang/miniconda3/envs/disag12/bin:\$PATH $PYTHON $BENCH \
                    --rank $rank --world-size $WORLD_SIZE --backend $backend --msg-bytes $sz \
                    --iters $ITERS --warmup $WARMUP --pipeline-depth $PIPELINE_DEPTH \
                    --master-addr $RECEIVER_IP --master-port $MASTER_PORT \
                    --ifname $IFNAME --all-ips $ALL_IPS --nixl-port $NIXL_PORT \
                    --out $OUT" &
                PIDS+=($!)
            fi
        done

        for pid in "${PIDS[@]}"; do
            wait "$pid" || true
        done

        echo ""
    done
done

for host in sgpu1 sgpu3 sgpu4 sgpu5 sgpu7 sgpu8 sgpu9; do
    rsync -av "$host:$OUTDIR/" "$OUTDIR/" 2>&1 | tail -2 || true
done

echo "=== Generating fan-in 7->1 plots ==="
$PYTHON "$SCRIPT_DIR/plot_fanin.py" --results-dir "$OUTDIR" --out-dir "$SCRIPT_DIR/plots"
echo "Done."
