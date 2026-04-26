#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/yizhuoliang/miniconda3/envs/disag12/bin/python"
BENCH="$SCRIPT_DIR/bench.py"

LOCAL_HOST="sgpu6"
REMOTE_HOST="sgpu7"
LOCAL_IP="10.0.0.5"
REMOTE_IP="10.0.0.6"
IFNAME="ens1f1np1"

ITERS=500
WARMUP=20
PIPELINE_DEPTH=16
OUTDIR="$SCRIPT_DIR/results"
mkdir -p "$OUTDIR"

MSG_SIZES=(65536 131072 262144 524288)
BACKENDS=(nccl nixl)

for backend in "${BACKENDS[@]}"; do
    for sz in "${MSG_SIZES[@]}"; do
        echo "=== $backend / ${sz}B ==="

        SENDER_OUT="$OUTDIR/${backend}_${sz}B_sender.json"
        RECVER_OUT="$OUTDIR/${backend}_${sz}B_receiver.json"

        MASTER_PORT=$((32000 + RANDOM % 1000))
        NIXL_PORT=$((15000 + RANDOM % 1000))

        $PYTHON "$BENCH" \
            --role receiver --backend "$backend" --msg-bytes "$sz" \
            --iters "$ITERS" --warmup "$WARMUP" --pipeline-depth "$PIPELINE_DEPTH" \
            --master-addr "$LOCAL_IP" --ifname "$IFNAME" \
            --local-ip "$LOCAL_IP" --remote-ip "$REMOTE_IP" \
            --master-port "$MASTER_PORT" --nixl-port "$NIXL_PORT" \
            --out "$RECVER_OUT" &
        LOCAL_PID=$!
        sleep 2

        ssh "$REMOTE_HOST" "PATH=/home/yizhuoliang/miniconda3/envs/disag12/bin:\$PATH $PYTHON $BENCH \
            --role sender --backend $backend --msg-bytes $sz \
            --iters $ITERS --warmup $WARMUP --pipeline-depth $PIPELINE_DEPTH \
            --master-addr $LOCAL_IP --ifname $IFNAME \
            --local-ip $REMOTE_IP --remote-ip $LOCAL_IP \
            --master-port $MASTER_PORT --nixl-port $NIXL_PORT \
            --out $SENDER_OUT"

        wait $LOCAL_PID || true

        echo ""
    done
done

rsync -av "$REMOTE_HOST:$OUTDIR/" "$OUTDIR/" 2>&1 | tail -3 || true

echo "=== Generating plots ==="
$PYTHON "$SCRIPT_DIR/plot.py" --results-dir "$OUTDIR" --out-dir "$SCRIPT_DIR/plots"
echo "Done."
