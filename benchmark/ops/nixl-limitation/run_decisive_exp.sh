#!/bin/bash
# run_decisive_exp.sh
#
# Runs the existing 10-request engine workload on the sphere-16 cluster but with
# the instrumented NIXL UCX backend (NIXL_UCX_POST_TRACE=1 + LD_PRELOAD of
# /tmp/nixl-repo/build-trace-wheelucx libs). Captures backend trace lines from
# the head-node server and from every Ray worker on every node, then runs the
# parser to determine where the 5ms post_xfer_req lives.
#
# This is the experiment FINDINGS_MAY6.md called for but never executed.
#
# Output dir: $REPO/benchmark/ops/nixl-limitation/results/decisive/<timestamp>/
#   server.log
#   worker_logs/<host>/*               (ray worker stderr files)
#   nixl_ucx_post_trace.log            (all backend trace lines, merged)
#   traces/<host>/...                  (DisagMoE advanced logging JSONs)
#   analysis.txt                       (parser output: substage breakdown +
#                                        engine vs backend correlation + verdict)

set -uo pipefail

REPO=/home/yizhuoliang/DisagMoE
NODES=(sgpu1 sgpu3 sgpu4 sgpu5 sgpu7 sgpu8 sgpu9)

TRACE_BUILD=/tmp/nixl-repo/build-trace-wheelucx
NIXL_BUNDLED_LIBS=/home/yizhuoliang/miniconda3/envs/disag12/lib/python3.12/site-packages/nixl_cu12.libs

DISAG_NIXL_FIX_MODE=${DISAG_NIXL_FIX_MODE:-per-call}
TAG=${TAG:-${DISAG_NIXL_FIX_MODE}}

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$REPO/benchmark/ops/nixl-limitation/results/decisive/run_${TS}_${TAG}"
mkdir -p "$OUTDIR/traces" "$OUTDIR/worker_logs"

source /home/yizhuoliang/miniconda3/etc/profile.d/conda.sh
conda activate disag12

if [ ! -f "$TRACE_BUILD/src/core/libnixl.so" ]; then
    echo "ERROR: instrumented NIXL build missing at $TRACE_BUILD on head node"
    exit 2
fi

echo "[0/6] Verifying instrumented build present on all nodes..."
MISSING=0
for n in "${NODES[@]}"; do
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" \
            "test -f $TRACE_BUILD/src/core/libnixl.so" 2>/dev/null; then
        echo "  $n: MISSING $TRACE_BUILD/src/core/libnixl.so"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "ERROR: Some nodes missing instrumented build. Rsync first:"
    echo "  for n in ${NODES[*]}; do ssh \$n mkdir -p /tmp/nixl-repo;"
    echo "    rsync -az $TRACE_BUILD/ \$n:$TRACE_BUILD/; done"
    exit 3
fi

NIXL_LD_PRELOAD="$TRACE_BUILD/src/bindings/libnixl_capi.so:$TRACE_BUILD/src/core/libnixl.so"
NIXL_LD_LIBRARY_PATH="$TRACE_BUILD/src/bindings:$TRACE_BUILD/src/core:$TRACE_BUILD/src/infra:$TRACE_BUILD/src/utils/common:$TRACE_BUILD/src/utils/serdes:$TRACE_BUILD/src/utils/stream:$TRACE_BUILD/src/utils/file:$TRACE_BUILD/subprojects/abseil-cpp-20250814.1:$TRACE_BUILD/subprojects/prometheus-cpp:$NIXL_BUNDLED_LIBS"

cat <<EOF | tee "$OUTDIR/env.txt"
TRACE_BUILD=$TRACE_BUILD
LD_PRELOAD=$NIXL_LD_PRELOAD
NIXL_PLUGIN_DIR=$TRACE_BUILD/src/plugins/ucx
NIXL_UCX_POST_TRACE=1
DISAG_NIXL_FIX_MODE=$DISAG_NIXL_FIX_MODE
EOF

echo "[1/6] Killing stale processes..."
pkill -9 -f benchmark/server.py 2>/dev/null
for n in "${NODES[@]}"; do
    ssh "$n" "pkill -9 -f benchmark/server.py 2>/dev/null; pkill -9 -f 'ray::' 2>/dev/null" &
done; wait
ray stop --force 2>/dev/null
for n in "${NODES[@]}"; do
    ssh "$n" "source /home/yizhuoliang/miniconda3/etc/profile.d/conda.sh && \
              conda activate disag12 && ray stop --force 2>/dev/null" &
done; wait
sleep 25

echo "[2/6] Starting Ray cluster with instrumented NIXL env (FIX_MODE=$DISAG_NIXL_FIX_MODE)..."
LD_PRELOAD="$NIXL_LD_PRELOAD" \
LD_LIBRARY_PATH="$NIXL_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
NIXL_PLUGIN_DIR="$TRACE_BUILD/src/plugins/ucx" \
NIXL_UCX_POST_TRACE=1 \
DISAG_NIXL_FIX_MODE="$DISAG_NIXL_FIX_MODE" \
ray start --head --node-ip-address=10.0.0.5 --port=6379 \
    --dashboard-port=8265 --min-worker-port=30000 --max-worker-port=39999 \
    --disable-usage-stats > "$OUTDIR/ray_head.log" 2>&1

for n in "${NODES[@]}"; do
    ssh "$n" "source /home/yizhuoliang/miniconda3/etc/profile.d/conda.sh && \
              conda activate disag12 && \
              LD_PRELOAD='$NIXL_LD_PRELOAD' \
              LD_LIBRARY_PATH='$NIXL_LD_LIBRARY_PATH' \
              NIXL_PLUGIN_DIR='$TRACE_BUILD/src/plugins/ucx' \
              NIXL_UCX_POST_TRACE=1 \
              DISAG_NIXL_FIX_MODE='$DISAG_NIXL_FIX_MODE' \
              ray start --address=10.0.0.5:6379 --disable-usage-stats" \
        > "$OUTDIR/worker_logs/${n}_ray_start.log" 2>&1 &
done; wait
sleep 25
ray status | tee "$OUTDIR/ray_status.txt"

echo "[3/6] Launching server with instrumented NIXL env (FIX_MODE=$DISAG_NIXL_FIX_MODE)..."
nohup env \
    LD_PRELOAD="$NIXL_LD_PRELOAD" \
    LD_LIBRARY_PATH="$NIXL_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
    NIXL_PLUGIN_DIR="$TRACE_BUILD/src/plugins/ucx" \
    NIXL_UCX_POST_TRACE=1 \
    DISAG_NIXL_FIX_MODE="$DISAG_NIXL_FIX_MODE" \
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
      --enable-advanced-logging \
      --advanced-logging-dir "$OUTDIR/traces" \
      --advanced-logging-sample-rate 1.0 \
    > "$OUTDIR/server.log" 2>&1 &

echo "Waiting for server up (Running on all addresses)..."
for i in $(seq 1 60); do
    if grep -q "Running on all addresses" "$OUTDIR/server.log" 2>/dev/null; then break; fi
    sleep 10
done

echo "[4/6] Running benchmark: 10 reqs (1 rps x 10s)..."
RATE=1
T=10
MAX_TIME=300
curl -s -o "$OUTDIR/result.json" \
    -X POST "http://localhost:6699/run_once" \
    -H "Content-Type: application/json" \
    -d "{\"rate\":$RATE,\"time\":$T,\"distribution\":\"poisson\",
         \"min_input_len\":256,\"max_input_len\":512,
         \"min_output_len\":256,\"max_output_len\":512}" \
    --max-time $MAX_TIME

echo "[5/6] Triggering trace dump and collecting from workers..."
pkill -USR1 -f 'ray::Engine' 2>/dev/null
for n in "${NODES[@]}"; do
    ssh "$n" "pkill -USR1 -f 'ray::Engine' 2>/dev/null" &
done; wait
sleep 25
for n in "${NODES[@]}"; do
    rsync -az "$n:$OUTDIR/traces/" "$OUTDIR/traces/" 2>/dev/null &
done; wait

echo "[5b/6] Harvesting Ray worker stderr (where NIXL_UCX_POST_TRACE lives)..."
for n in "${NODES[@]}"; do
    ssh "$n" "ls /tmp/ray/session_latest/logs/worker-*.err 2>/dev/null" \
        > "$OUTDIR/worker_logs/${n}_worker_files.txt" 2>/dev/null &
done; wait
for n in "${NODES[@]}"; do
    mkdir -p "$OUTDIR/worker_logs/${n}"
    rsync -az --include='*.err' --include='*.out' --exclude='*' \
        "$n:/tmp/ray/session_latest/logs/" "$OUTDIR/worker_logs/${n}/" 2>/dev/null &
done; wait
mkdir -p "$OUTDIR/worker_logs/head"
rsync -az --include='*.err' --include='*.out' --exclude='*' \
    /tmp/ray/session_latest/logs/ "$OUTDIR/worker_logs/head/" 2>/dev/null

echo "[5c/6] Merging NIXL_UCX_POST_TRACE lines..."
{
    grep "NIXL_UCX_POST_TRACE" "$OUTDIR/server.log" 2>/dev/null || true
    find "$OUTDIR/worker_logs" -type f \( -name '*.err' -o -name '*.out' -o -name '*.log' \) \
        -print0 | xargs -0 grep -h "NIXL_UCX_POST_TRACE" 2>/dev/null || true
} > "$OUTDIR/nixl_ucx_post_trace.log"

NTRACE=$(wc -l < "$OUTDIR/nixl_ucx_post_trace.log")
echo "Captured $NTRACE NIXL_UCX_POST_TRACE lines"

echo "[6/6] Running parser..."
python "$REPO/benchmark/ops/nixl-limitation/parse_backend_traces.py" \
    "$OUTDIR/nixl_ucx_post_trace.log" \
    --traces-dir "$OUTDIR/traces" \
    --threshold-us 1000 \
    --out "$OUTDIR/analysis.txt"

echo ""
echo "Done. Output dir: $OUTDIR"
echo "Quick stats from server.log:"
grep -E "Detokenizer.*ITL|^ITL" "$OUTDIR/server.log" | tail -5 || true
[ -f "$OUTDIR/result.json" ] && echo "result.json: $(cat $OUTDIR/result.json | head -c 400)"
echo ""
echo "OUTDIR=$OUTDIR" > "$REPO/benchmark/ops/nixl-limitation/.last_decisive"
