#!/bin/bash
set -uo pipefail
REPO=/home/yizhuoliang/DisagMoE
SCRIPT=$REPO/benchmark/ops/nixl-limitation/run_shared_agent_mesh.sh

ITERS=${ITERS:-1000}
WARMUP=${WARMUP:-100}
NUM_PEERS=${NUM_PEERS:-6}
INFLIGHT=${INFLIGHT:-32}
RANK1_INFLIGHT=${RANK1_INFLIGHT:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
DESC_LEN_BYTES=${DESC_LEN_BYTES:-4194304}

PAYLOADS=${PAYLOADS:-"8192 16384 32768 65536 131072"}
MODES=${MODES:-"current per-call log2-pool tight-fixed"}

PORT_BASE=${PORT_BASE:-2000}

TS=$(date +%Y%m%d_%H%M%S)
SWEEP_OUT=$REPO/benchmark/ops/nixl-limitation/results/shared_mesh/sweep_${TS}
mkdir -p "$SWEEP_OUT"
SUMMARY=$SWEEP_OUT/summary.csv
echo "fix_mode,desc_len,bytes_to_write,num_peers,inflight,r1if,iters,total_samples,make_p50_ns,make_p90_ns,make_p99_ns,post_p50_ns,post_p90_ns,post_p99_ns,post_max_ns,wall_total_s,throughput_per_s,run_dir" > "$SUMMARY"

PORT=$PORT_BASE
for MODE in $MODES; do
    for B in $PAYLOADS; do
        if [ "$MODE" = "tight-fixed" ]; then
            DLB=$B
            ACTUAL_BYTES=$B
        else
            DLB=$DESC_LEN_BYTES
            ACTUAL_BYTES=$B
        fi

        TAG="${MODE}_b${B}_dl${DLB}"
        echo "=========================================="
        echo "[$(date +%H:%M:%S)] sweep: mode=$MODE bytes=$B desc_len=$DLB port_offset=$PORT"
        echo "=========================================="

        FIX_MODE=$MODE \
        BYTES_TO_WRITE=$ACTUAL_BYTES \
        DESC_LEN_BYTES=$DLB \
        ITERS=$ITERS \
        WARMUP=$WARMUP \
        NUM_PEERS=$NUM_PEERS \
        INFLIGHT=$INFLIGHT \
        RANK1_INFLIGHT=$RANK1_INFLIGHT \
        NUM_WORKERS=$NUM_WORKERS \
        PORT_OFFSET=$PORT \
        TAG=$TAG \
            bash "$SCRIPT" 2>&1 | tail -5 | tee -a "$SWEEP_OUT/sweep.log"

        LATEST=$(ls -td $REPO/benchmark/ops/nixl-limitation/results/shared_mesh/run_*_${TAG}_* 2>/dev/null | head -1)
        if [ -n "$LATEST" ] && [ -f "$LATEST/rank0.log" ]; then
            RESULT_LINE=$(grep "^RESULT" "$LATEST/rank0.log" | head -1)
            if [ -n "$RESULT_LINE" ]; then
                FM=$(echo "$RESULT_LINE" | grep -o "fix_mode=[^,]*" | cut -d= -f2)
                DL=$(echo "$RESULT_LINE" | grep -o "desc_len=[^,]*" | cut -d= -f2)
                BW=$(echo "$RESULT_LINE" | grep -o "bytes_to_write=[^,]*" | cut -d= -f2)
                NP=$(echo "$RESULT_LINE" | grep -o "num_peers=[^,]*" | cut -d= -f2)
                IF=$(echo "$RESULT_LINE" | grep -o "inflight=[^,]*" | cut -d= -f2)
                IT=$(echo "$RESULT_LINE" | grep -o "iters=[^,]*" | cut -d= -f2)
                TS_=$(echo "$RESULT_LINE" | grep -o "total_samples=[^,]*" | cut -d= -f2)
                MP50=$(echo "$RESULT_LINE" | grep -o "make_p50_ns=[^,]*" | cut -d= -f2)
                MP90=$(echo "$RESULT_LINE" | grep -o "make_p90_ns=[^,]*" | cut -d= -f2)
                MP99=$(echo "$RESULT_LINE" | grep -o "make_p99_ns=[^,]*" | cut -d= -f2)
                PP50=$(echo "$RESULT_LINE" | grep -o "post_p50_ns=[^,]*" | cut -d= -f2)
                PP90=$(echo "$RESULT_LINE" | grep -o "post_p90_ns=[^,]*" | cut -d= -f2)
                PP99=$(echo "$RESULT_LINE" | grep -o "post_p99_ns=[^,]*" | cut -d= -f2)
                PMAX=$(echo "$RESULT_LINE" | grep -o "post_max_ns=[^,]*" | cut -d= -f2)
                WT=$(echo "$RESULT_LINE" | grep -o "wall_total_s=[^,]*" | cut -d= -f2)
                TP=$(echo "$RESULT_LINE" | grep -o "throughput_per_s=[^,]*" | cut -d= -f2)
                echo "$FM,$DL,$BW,$NP,$IF,$RANK1_INFLIGHT,$IT,$TS_,$MP50,$MP90,$MP99,$PP50,$PP90,$PP99,$PMAX,$WT,$TP,$LATEST" >> "$SUMMARY"
                echo "  -> p50_post=${PP50}ns p99_post=${PP99}ns thr=${TP}/s"
            else
                echo "  WARN: no RESULT line in $LATEST/rank0.log"
            fi
        fi

        sleep 3
        PORT=$((PORT + 10))
    done
done

echo ""
echo "==========================================  "
echo "Sweep complete. Summary at: $SUMMARY"
echo "=========================================="
column -s, -t "$SUMMARY" | head -50
