#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/yizhuoliang/miniconda3/envs/disag12/bin/python"
BENCH="$SCRIPT_DIR/bench.py"

LOCAL_HOST="sgpu6"
REMOTE_HOST="sgpu7"
LOCAL_IP="10.0.0.1"
REMOTE_IP="10.0.0.2"
IFNAME="ens1f1np1"

ITERS=100
WARMUP=20
OUTDIR="$SCRIPT_DIR/results"
mkdir -p "$OUTDIR"

# batch_size * hidden_size(4096) * 2(bf16) for GLM-4.5-Air-106B
MSG_SIZES=(131072 262144 524288)
BACKENDS=(nccl nixl)

for backend in "${BACKENDS[@]}"; do
    for sz in "${MSG_SIZES[@]}"; do
        echo "=== $backend / ${sz}B ==="

        SENDER_OUT="$OUTDIR/${backend}_${sz}B_sender.json"
        RECVER_OUT="$OUTDIR/${backend}_${sz}B_receiver.json"

        NCCL_PORT=$((32000 + RANDOM % 1000))

        if [ "$backend" = "nccl" ]; then
            $PYTHON "$BENCH" \
                --role receiver --backend nccl --msg-bytes "$sz" \
                --iters "$ITERS" --warmup "$WARMUP" \
                --master-addr "$LOCAL_IP" --ifname "$IFNAME" \
                --master-port "$NCCL_PORT" \
                --out "$RECVER_OUT" &
            LOCAL_PID=$!
            sleep 2

            ssh "$REMOTE_HOST" "NCCL_SOCKET_IFNAME=$IFNAME NCCL_IB_HCA=mlx5_1 \
                $PYTHON $BENCH \
                --role sender --backend nccl --msg-bytes $sz \
                --iters $ITERS --warmup $WARMUP \
                --master-addr $LOCAL_IP --ifname $IFNAME \
                --master-port $NCCL_PORT \
                --out $SENDER_OUT"

            wait $LOCAL_PID || true
        else
            NIXL_PORT=15000

            ssh "$REMOTE_HOST" "UCX_NET_DEVICES=mlx5_1:1 \
                $PYTHON $BENCH \
                --role receiver --backend nixl --msg-bytes $sz \
                --iters $ITERS --warmup $WARMUP \
                --local-ip $REMOTE_IP --remote-ip $LOCAL_IP \
                --nixl-port $NIXL_PORT \
                --out $RECVER_OUT" &
            REMOTE_PID=$!
            sleep 3

            UCX_NET_DEVICES=mlx5_1:1 $PYTHON "$BENCH" \
                --role sender --backend nixl --msg-bytes "$sz" \
                --iters "$ITERS" --warmup "$WARMUP" \
                --local-ip "$LOCAL_IP" --remote-ip "$REMOTE_IP" \
                --nixl-port "$NIXL_PORT" \
                --out "$SENDER_OUT"

            wait $REMOTE_PID || true
        fi

        echo ""
    done
done

echo "=== Generating plots ==="
$PYTHON "$SCRIPT_DIR/plot.py" --results-dir "$OUTDIR" --out-dir "$SCRIPT_DIR/plots"
echo "Done."
