#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/yizhuoliang/miniconda3/envs/disag12/bin/python"
BENCH="$SCRIPT_DIR/bench_fanin.py"

HOSTS=(sgpu6 sgpu7 sgpu8 sgpu9)
ALL_IPS="10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4"
RECEIVER_IP="10.0.0.1"
IFNAME="ens1f1np1"

ITERS=100
WARMUP=20
OUTDIR="$SCRIPT_DIR/results_fanin"
mkdir -p "$OUTDIR"

# batch_size * hidden_size(4096) * 2(bf16) for GLM-4.5-Air-106B
MSG_SIZES=(131072 262144 524288)
BACKENDS=(nccl nixl)

for backend in "${BACKENDS[@]}"; do
    for sz in "${MSG_SIZES[@]}"; do
        echo "=== $backend / $(numfmt --to=iec $sz) / fan-in 3→1 ==="

        MASTER_PORT=$((31000 + RANDOM % 1000))
        NIXL_PORT=$((16000 + RANDOM % 1000))
        PIDS=()

        for rank in 0 1 2 3; do
            host="${HOSTS[$rank]}"
            OUT="$OUTDIR/${backend}_${sz}B_rank${rank}.json"

            if [ "$rank" -eq 0 ]; then
                $PYTHON "$BENCH" \
                    --rank $rank --world-size 4 --backend "$backend" --msg-bytes "$sz" \
                    --iters "$ITERS" --warmup "$WARMUP" \
                    --master-addr "$RECEIVER_IP" --master-port "$MASTER_PORT" \
                    --ifname "$IFNAME" --all-ips "$ALL_IPS" --nixl-port "$NIXL_PORT" \
                    --out "$OUT" &
                PIDS+=($!)
            else
                ssh "$host" "$PYTHON $BENCH \
                    --rank $rank --world-size 4 --backend $backend --msg-bytes $sz \
                    --iters $ITERS --warmup $WARMUP \
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

for host in sgpu7 sgpu8 sgpu9; do
    rsync -q "$host:$OUTDIR/" "$OUTDIR/" 2>/dev/null || true
done

echo "=== Generating fan-in plots ==="
$PYTHON "$SCRIPT_DIR/plot_fanin.py" --results-dir "$OUTDIR" --out-dir "$SCRIPT_DIR/plots"
echo "Done."
