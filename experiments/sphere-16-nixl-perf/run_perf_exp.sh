#!/bin/bash
# Runs one DisagMoE engine + benchmark with a selected transport (nccl|nixl) and
# load profile (RATE req/s for TIME_SEC seconds). Captures:
#   - server.log (stdout + Detokenizer ITL stream + final Metrics: line)
#   - result.json (HTTP body of /run_once, or empty if curl timed out)
#   - env.txt, ray_status.txt
# Output goes to experiments/sphere-16-nixl-perf/results/<TAG>_<TS>/
#
# Required env: TRANSPORT (nccl|nixl), RATE (int), TIME_SEC (int), TAG (label)
# Optional env: MAX_CURL_TIME (curl --max-time, default 1800),
#               MIN_IN_LEN / MAX_IN_LEN (default 256/512),
#               MIN_OUT_LEN / MAX_OUT_LEN (default 256/512)

set -uo pipefail
REPO=/home/yizhuoliang/DisagMoE
NODES=(sgpu1 sgpu3 sgpu4 sgpu5 sgpu7 sgpu8 sgpu9)

TRANSPORT=${TRANSPORT:?TRANSPORT=nccl|nixl required}
RATE=${RATE:?RATE=int required}
TIME_SEC=${TIME_SEC:?TIME_SEC=int required}
TAG=${TAG:-${TRANSPORT}_rps${RATE}_t${TIME_SEC}}
MAX_CURL_TIME=${MAX_CURL_TIME:-1800}
MIN_IN_LEN=${MIN_IN_LEN:-256}
MAX_IN_LEN=${MAX_IN_LEN:-512}
MIN_OUT_LEN=${MIN_OUT_LEN:-256}
MAX_OUT_LEN=${MAX_OUT_LEN:-512}

case "$TRANSPORT" in
    nccl|nixl) ;;
    *) echo "TRANSPORT must be nccl or nixl"; exit 2;;
esac

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$REPO/experiments/sphere-16-nixl-perf/results/${TAG}_${TS}"
mkdir -p "$OUTDIR/traces" "$OUTDIR/worker_logs"

source /home/yizhuoliang/miniconda3/etc/profile.d/conda.sh
conda activate disag12

ACTIVE_SO=/home/yizhuoliang/DisagMoE/disagmoe_c.cpython-312-x86_64-linux-gnu.so
BACKUP_SO=/tmp/disagmoe_c.${TRANSPORT}.so

echo "==> swapping active .so to ${TRANSPORT} variant on every node"
if [ ! -f "$BACKUP_SO" ]; then echo "ERROR: $BACKUP_SO not on head"; exit 3; fi
cp "$BACKUP_SO" "$ACTIVE_SO"
for n in "${NODES[@]}"; do
    (timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 -o ServerAliveInterval=2 -o ServerAliveCountMax=2 \
        "$n" "test -f $BACKUP_SO && cp $BACKUP_SO $ACTIVE_SO && echo $n: ok" 2>&1) &
done
wait

TRACE_BUILD=/tmp/nixl-repo/build-trace-wheelucx
NIXL_BUNDLED_LIBS=/home/yizhuoliang/miniconda3/envs/disag12/lib/python3.12/site-packages/nixl_cu12.libs
NIXL_LD_LIBRARY_PATH="$TRACE_BUILD/src/bindings:$TRACE_BUILD/src/core:$TRACE_BUILD/src/infra:$TRACE_BUILD/src/utils/common:$TRACE_BUILD/src/utils/serdes:$TRACE_BUILD/src/utils/stream:$TRACE_BUILD/src/utils/file:$TRACE_BUILD/subprojects/abseil-cpp-20250814.1:$TRACE_BUILD/subprojects/prometheus-cpp:$NIXL_BUNDLED_LIBS"

EXTRA_RUN_ENV=()
EFFECTIVE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ "$TRANSPORT" = "nixl" ]; then
    NIXL_LD_PRELOAD="$TRACE_BUILD/src/bindings/libnixl_capi.so:$TRACE_BUILD/src/core/libnixl.so"
    EXTRA_RUN_ENV+=( "LD_PRELOAD=$NIXL_LD_PRELOAD" )
    EXTRA_RUN_ENV+=( "NIXL_PLUGIN_DIR=$TRACE_BUILD/src/plugins/ucx" )
    EXTRA_RUN_ENV+=( "NIXL_UCX_POST_TRACE=0" )
    EFFECTIVE_LD_LIBRARY_PATH="$NIXL_LD_LIBRARY_PATH:$EFFECTIVE_LD_LIBRARY_PATH"
fi

cat <<EOF | tee "$OUTDIR/env.txt"
TRANSPORT=$TRANSPORT
RATE=$RATE
TIME_SEC=$TIME_SEC
TAG=$TAG
ACTIVE_SO=$ACTIVE_SO (size=$(stat -c %s $ACTIVE_SO))
MIN_IN_LEN=$MIN_IN_LEN MAX_IN_LEN=$MAX_IN_LEN
MIN_OUT_LEN=$MIN_OUT_LEN MAX_OUT_LEN=$MAX_OUT_LEN
MAX_CURL_TIME=$MAX_CURL_TIME
EOF

echo "==> cleaning up stale procs"
pkill -9 -f benchmark/server.py 2>/dev/null
pkill -9 -f 'ray::' 2>/dev/null
pkill -9 -f raylet 2>/dev/null
for n in "${NODES[@]}"; do
    (timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=3 "$n" \
        "pkill -9 -f benchmark/server.py 2>/dev/null; pkill -9 -f 'ray::' 2>/dev/null; pkill -9 -f raylet 2>/dev/null; echo $n: cleaned" 2>&1) &
done
wait
ray stop --force >/dev/null 2>&1
for n in "${NODES[@]}"; do
    (timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=3 "$n" \
        "source /home/yizhuoliang/miniconda3/etc/profile.d/conda.sh && conda activate disag12 && ray stop --force >/dev/null 2>&1; echo $n: ray stopped" 2>&1) &
done
wait
sleep 8

echo "==> starting Ray cluster (transport=$TRANSPORT)"
LD_LIBRARY_ARG=""
if [ "$TRANSPORT" = "nixl" ] && [ -n "$EFFECTIVE_LD_LIBRARY_PATH" ]; then
    LD_LIBRARY_ARG="LD_LIBRARY_PATH=$EFFECTIVE_LD_LIBRARY_PATH"
fi

env "${EXTRA_RUN_ENV[@]}" $LD_LIBRARY_ARG \
    ray start --head --node-ip-address=10.0.0.5 --port=6379 \
    --dashboard-port=8265 --min-worker-port=30000 --max-worker-port=39999 \
    --disable-usage-stats > "$OUTDIR/ray_head.log" 2>&1

EXTRA_SSH_ENV=""
for kv in "${EXTRA_RUN_ENV[@]}"; do
    EXTRA_SSH_ENV+=" $kv "
done
SSH_LD_ARG=""
if [ "$TRANSPORT" = "nixl" ]; then
    SSH_LD_ARG="LD_LIBRARY_PATH='$NIXL_LD_LIBRARY_PATH'"
fi

for n in "${NODES[@]}"; do
    (timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=5 "$n" \
        "source /home/yizhuoliang/miniconda3/etc/profile.d/conda.sh && \
         conda activate disag12 && \
         env $EXTRA_SSH_ENV $SSH_LD_ARG \
             ray start --address=10.0.0.5:6379 --disable-usage-stats" \
        > "$OUTDIR/worker_logs/${n}_ray_start.log" 2>&1) &
done
wait
sleep 18
ray status > "$OUTDIR/ray_status.txt" 2>&1

echo "==> launching server (transport=$TRANSPORT rate=$RATE t=$TIME_SEC)"
nohup env "${EXTRA_RUN_ENV[@]}" $LD_LIBRARY_ARG \
    NCCL_RUNTIME_CONNECT=0 SKIP_WARMUP=1 \
    python "$REPO/benchmark/server.py" \
      -N 8 -g 2 -u 0.90 \
      --model glm45air_106b --attn-qkv-quant none --moe-linear-quant none \
      --max-batch-size-attn 256 --max-attn-graph-bsz 256 --max-pending-sends 16 \
      --max-batch-size-expert 1024 --block-size 16 \
      --placement colocate --dp-size 16 --ep-size 16 \
      --transport zmq --host-ifname ens1f1np1 \
      --nccl-ib-hca mlx5_1 --nccl-ib-gid-index 3 \
      --unified-scheduler-type defrag --defrag-weight-decay 0.8 \
      --defrag-lookahead-steps 4 --defrag-lookback-steps 4 \
      --less-than-sm90 --cuda-graph-attn --cuda-graph-expert \
      --analyze-throughput \
      --gate-profile-file "$REPO/gating_profiles/glm45air_gating_profiles/gating_glm45air_sharegpt_200.parquet" \
      --num-shared-experts 1 --shared-expert-intermediate-size 1408 \
    > "$OUTDIR/server.log" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

echo "==> waiting for server up (max 1500s)"
for i in $(seq 1 100); do
    if grep -q "Running on all addresses\|Running on http://0.0.0.0" "$OUTDIR/server.log" 2>/dev/null; then
        echo "  server up at $(date +%H:%M:%S) after $((i*15))s"
        break
    fi
    if ! pgrep -f benchmark/server.py >/dev/null 2>&1; then
        echo "ERROR: server died early"; tail -30 "$OUTDIR/server.log"; exit 4
    fi
    if [ $((i % 4)) -eq 0 ]; then
        echo "  waiting ${i}x15s; last warmup line:"
        grep -E "warmup|Now running|put .* requests" "$OUTDIR/server.log" 2>/dev/null | tail -1
    fi
    sleep 15
done

if ! grep -q "Running on all addresses\|Running on http://0.0.0.0" "$OUTDIR/server.log" 2>/dev/null; then
    echo "ERROR: server did not come up. tail of server.log:"
    tail -40 "$OUTDIR/server.log"
    exit 4
fi

sleep 5
echo "==> running benchmark: rate=$RATE rps × time=${TIME_SEC}s (curl max_time=$MAX_CURL_TIME)"
START_TS=$(date +%s)
curl -s -o "$OUTDIR/result.json" \
    -X POST "http://localhost:6699/run_once" \
    -H "Content-Type: application/json" \
    -d "{\"rate\":$RATE,\"time\":$TIME_SEC,\"distribution\":\"poisson\",
         \"min_input_len\":$MIN_IN_LEN,\"max_input_len\":$MAX_IN_LEN,
         \"min_output_len\":$MIN_OUT_LEN,\"max_output_len\":$MAX_OUT_LEN}" \
    --max-time $MAX_CURL_TIME
END_TS=$(date +%s)
echo "curl finished after $((END_TS - START_TS))s"

echo "==> giving Metrics: line a chance to flush (sleep 10)"
sleep 10

echo "==> stopping server + ray"
kill -9 $SERVER_PID 2>/dev/null
pkill -9 -f benchmark/server.py 2>/dev/null
pkill -9 -f 'ray::' 2>/dev/null
for n in "${NODES[@]}"; do
    (timeout 8 ssh -o BatchMode=yes "$n" "pkill -9 -f benchmark/server.py 2>/dev/null; pkill -9 -f 'ray::' 2>/dev/null; pkill -9 -f raylet 2>/dev/null" 2>&1) &
done
wait
ray stop --force >/dev/null 2>&1

echo ""
echo "==> Captured Metrics from server.log:"
echo "----- BEGIN METRICS -----"
awk '/^Metrics:/{flag=1} flag{print} /^peak_itl_p99/{flag=0}' "$OUTDIR/server.log" 2>/dev/null | head -20
echo "----- END METRICS -----"

echo ""
echo "==> ITL stream (last 10 Detokenizer prints):"
grep "ITL mean" "$OUTDIR/server.log" 2>/dev/null | sed -E 's/.*\)([0-9.]+) - .*token throughput: ([0-9.]+)k.*ITL mean=([0-9.]+)ms p50=([0-9.]+)ms p99=([0-9.]+)ms/t=\1 throughput=\2k mean=\3 p50=\4 p99=\5/' | tail -10

echo ""
echo "==> Result.json:"
cat "$OUTDIR/result.json" 2>/dev/null | head -c 1200
echo ""

echo "OUTDIR=$OUTDIR"
echo "DONE_$TAG"
