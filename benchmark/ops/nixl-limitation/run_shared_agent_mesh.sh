#!/bin/bash
set -uo pipefail
REPO=/home/yizhuoliang/DisagMoE
BIN=$REPO/benchmark/ops/nixl-limitation/shared_agent_mesh
HEAD_IP=${HEAD_IP:-10.0.0.5}
NODES=(sgpu3 sgpu4 sgpu5 sgpu7 sgpu8 sgpu9)
NUM_PEERS=${NUM_PEERS:-${#NODES[@]}}

TRACE_BUILD=/tmp/nixl-repo/build-trace-wheelucx
NIXL_BUNDLED_LIBS=/home/yizhuoliang/miniconda3/envs/disag12/lib/python3.12/site-packages/nixl_cu12.libs
NIXL_LD_PATH="$TRACE_BUILD/src/bindings:$TRACE_BUILD/src/core:$TRACE_BUILD/src/infra:$TRACE_BUILD/src/utils/common:$TRACE_BUILD/src/utils/serdes:$TRACE_BUILD/src/utils/stream:$TRACE_BUILD/src/utils/file:$TRACE_BUILD/subprojects/abseil-cpp-20250814.1:$TRACE_BUILD/subprojects/prometheus-cpp:$NIXL_BUNDLED_LIBS"

PORT_OFFSET=${PORT_OFFSET:-100}
INFLIGHT=${INFLIGHT:-32}
RANK1_INFLIGHT=${RANK1_INFLIGHT:-0}
ITERS=${ITERS:-3000}
WARMUP=${WARMUP:-300}
MEM=${MEM:-vram}
NUM_WORKERS=${NUM_WORKERS:-4}
NIXL_TRACE=${NIXL_TRACE:-0}

# [FIX MODE EXTENSIONS]
FIX_MODE=${FIX_MODE:-current}
BYTES_TO_WRITE=${BYTES_TO_WRITE:-}                # single payload (bytes), optional
BYTES_TO_WRITE_LIST=${BYTES_TO_WRITE_LIST:-}      # comma-separated payload sizes, optional
DESC_LEN_BYTES=${DESC_LEN_BYTES:-4194304}         # default 4 MiB to match engine
POOL_SIZES=${POOL_SIZES:-}                        # for log2-pool, optional

TS=$(date +%Y%m%d_%H%M%S)
TAG=${TAG:-${FIX_MODE}}
OUT=$REPO/benchmark/ops/nixl-limitation/results/shared_mesh/run_${TS}_${TAG}_n${NUM_PEERS}_if${INFLIGHT}_r1if${RANK1_INFLIGHT}
mkdir -p "$OUT"
echo "OUT=$OUT  NUM_PEERS=$NUM_PEERS  FIX_MODE=$FIX_MODE  BYTES=$BYTES_TO_WRITE($BYTES_TO_WRITE_LIST)"

PEER_HOST_LIST=""
for ((i=0; i<NUM_PEERS; i++)); do
    PEER_HOST_LIST+=$HEAD_IP
    [ $((i+1)) -lt $NUM_PEERS ] && PEER_HOST_LIST+=","
done

for ((p=0; p<NUM_PEERS; p++)); do
    NODE=${NODES[$p]}
    rsync -a "$BIN" "$NODE:$BIN" >/dev/null &
done
wait

SSH_PIDS=()
for ((p=0; p<NUM_PEERS; p++)); do
    NODE=${NODES[$p]}
    REMOTE_CMD="set -e; \
      export LD_LIBRARY_PATH=$NIXL_LD_PATH:\${LD_LIBRARY_PATH:-}; \
      export NIXL_PLUGIN_DIR=$TRACE_BUILD/src/plugins/ucx; \
      export NIXL_UCX_POST_TRACE=$NIXL_TRACE; \
      export UCX_LOG_LEVEL=warn; \
      cd $REPO/benchmark/ops/nixl-limitation; \
      CUDA_VISIBLE_DEVICES=0 ./shared_agent_mesh \
        --rank 1 --peer-host $HEAD_IP --port-offset $PORT_OFFSET \
        --num-peers 1 --peer-idx $p \
        --num-workers $NUM_WORKERS \
        --rank1-inflight $RANK1_INFLIGHT \
        --desc-len-bytes $DESC_LEN_BYTES \
        --mem $MEM --cuda-device 0 \
        > /tmp/shamesh_rank1_p${p}.log 2>&1"
    ssh "$NODE" "$REMOTE_CMD" &
    SSH_PIDS+=($!)
    echo "  launched rank 1 peer_idx=$p on $NODE"
done

cleanup() {
    for ((p=0; p<NUM_PEERS; p++)); do
        NODE=${NODES[$p]}
        ( timeout 5 ssh -o BatchMode=yes -o ConnectTimeout=2 "$NODE" \
            "pkill -9 -f shared_agent_mesh" 2>/dev/null ) &
    done
    wait
    for pid in "${SSH_PIDS[@]}"; do kill -9 $pid 2>/dev/null; done
    for ((p=0; p<NUM_PEERS; p++)); do
        NODE=${NODES[$p]}
        scp "$NODE:/tmp/shamesh_rank1_p${p}.log" "$OUT/rank1_p${p}.log" 2>/dev/null &
    done
    wait
}
trap cleanup EXIT

sleep 12

export LD_LIBRARY_PATH=$NIXL_LD_PATH:${LD_LIBRARY_PATH:-}
export NIXL_PLUGIN_DIR=$TRACE_BUILD/src/plugins/ucx
export NIXL_UCX_POST_TRACE=$NIXL_TRACE
export UCX_LOG_LEVEL=warn

EXTRA_ARGS=()
EXTRA_ARGS+=(--fix-mode "$FIX_MODE")
EXTRA_ARGS+=(--desc-len-bytes "$DESC_LEN_BYTES")
if [ -n "$BYTES_TO_WRITE_LIST" ]; then
    EXTRA_ARGS+=(--bytes-to-write-list "$BYTES_TO_WRITE_LIST")
elif [ -n "$BYTES_TO_WRITE" ]; then
    EXTRA_ARGS+=(--bytes-to-write "$BYTES_TO_WRITE")
fi
if [ -n "$POOL_SIZES" ]; then
    EXTRA_ARGS+=(--pool-sizes "$POOL_SIZES")
fi

CUDA_VISIBLE_DEVICES=0 "$BIN" \
    --rank 0 --peer-host $HEAD_IP --port-offset $PORT_OFFSET \
    --num-peers $NUM_PEERS --inflight $INFLIGHT \
    --num-workers $NUM_WORKERS \
    --peer-host-list "$PEER_HOST_LIST" \
    --iters $ITERS --warmup $WARMUP \
    --mem $MEM --cuda-device 0 \
    --out-csv "$OUT/rank0_lat.csv" \
    "${EXTRA_ARGS[@]}" \
    > "$OUT/rank0.log" 2>&1

echo "=== RESULT ==="
grep "^RESULT" "$OUT/rank0.log" || { echo "no RESULT"; tail -30 "$OUT/rank0.log"; }
echo "OUT=$OUT"
