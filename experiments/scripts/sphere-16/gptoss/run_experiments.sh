#!/usr/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Sphere-16 Experiment Runner
# Runs 8 experiments: 4 gate profiles × {logging off, logging on}
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$HOME/DisagMoE"
RESULTS_DIR="${REPO_DIR}/experiments/scripts/sphere-16/results"
CONDA_DIR="$HOME/miniconda3"

# ── Fixed server config (matches launch_server.sh) ──────────
N_NODE=8
N_GPU_PER_NODE=2
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))
MODEL_NAME="gptoss_120b"
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"
PLACEMENT="colocate"
TRANSPORT="zmq"
HOST_IFNAME="ens1f1np1"
NCCL_IB_HCA="mlx5_1"
NCCL_IB_GID_INDEX="3"
MAX_BATCH_SIZE_ATTN=256
MAX_BATCH_SIZE_EXP=512
MAX_PENDING_SENDS=16
UNIFIED_SCHEDULER_TYPE="defrag"
DEFRAG_WEIGHT_DECAY=0.8
DEFRAG_LOOKAHEAD_STEPS=4
DEFRAG_LOOKBACK_STEPS=4
ADV_LOG_SAMPLE_RATE=0.1
SERVER_PORT=6699

# ── Benchmark request payload ───────────────────────────────
BENCH_PAYLOAD='{
    "rate": 2000,
    "time": 4,
    "distribution": "poisson",
    "min_input_len": 64,
    "max_input_len": 128,
    "min_output_len": 256,
    "max_output_len": 512
}'

# ── Profiles to test ────────────────────────────────────────
PROFILES=(
    "gating_chinese_zhihu_200.parquet:chinese_zhihu"
    "gating_gptoss120b_sharegpt_200.parquet:sharegpt"
    "gating_legal_court_opinions_200.parquet:legal_court"
    "gating_math_gsm8k_200.parquet:math_gsm8k"
)

# ── Helper functions ─────────────────────────────────────────

log() { echo "$(date '+%H:%M:%S') [EXP] $*"; }

kill_server() {
    log "Killing existing server..."
    # Kill any python benchmark/server.py process
    pkill -f "python benchmark/server.py" 2>/dev/null || true
    # Also kill tmux session if it exists
    tmux kill-session -t disagmoe-server 2>/dev/null || true
    sleep 5
    # Verify it's dead
    if pgrep -f "python benchmark/server.py" >/dev/null 2>&1; then
        pkill -9 -f "python benchmark/server.py" 2>/dev/null || true
        sleep 3
    fi
    log "Server killed."
}

launch_server() {
    local gate_profile="$1"
    local adv_logging="$2"      # "on" or "off"
    local adv_log_dir="$3"      # only used when adv_logging=on
    local log_file="$4"

    local ADV_LOG_ARGS=""
    if [ "$adv_logging" = "on" ]; then
        mkdir -p "$adv_log_dir"
        ADV_LOG_ARGS="--enable-advanced-logging --advanced-logging-dir $adv_log_dir --advanced-logging-sample-rate $ADV_LOG_SAMPLE_RATE"
    fi

    log "Launching server: profile=$(basename $gate_profile) logging=$adv_logging"

    cd "$REPO_DIR"
    nohup env NCCL_RUNTIME_CONNECT="${NCCL_RUNTIME_CONNECT:-0}" python benchmark/server.py \
        -N $N_NODE \
        -g $N_GPU_PER_NODE \
        -u 0.98 \
        --model $MODEL_NAME \
        --attn-qkv-quant $ATTN_QKV_QUANT \
        --moe-linear-quant $MOE_LINEAR_QUANT \
        --max-batch-size-attn $MAX_BATCH_SIZE_ATTN \
        --max-attn-graph-bsz $MAX_BATCH_SIZE_ATTN \
        --max-pending-sends $MAX_PENDING_SENDS \
        --max-batch-size-exp $MAX_BATCH_SIZE_EXP \
        --block-size 16 \
        --placement $PLACEMENT \
        --dp-size $WORLD_SIZE \
        --ep-size $WORLD_SIZE \
        --transport $TRANSPORT \
        --host-ifname $HOST_IFNAME \
        --nccl-ib-hca $NCCL_IB_HCA \
        --nccl-ib-gid-index $NCCL_IB_GID_INDEX \
        --unified-scheduler-type $UNIFIED_SCHEDULER_TYPE \
        --defrag-weight-decay $DEFRAG_WEIGHT_DECAY \
        --defrag-lookahead-steps $DEFRAG_LOOKAHEAD_STEPS \
        --defrag-lookback-steps $DEFRAG_LOOKBACK_STEPS \
        --less-than-sm90 \
        --cuda-graph-attn \
        --cuda-graph-expert \
        --file "$RESULTS_DIR/benchmark.csv" \
        --analyze-throughput \
        --trace \
        --gate-profile-file "$gate_profile" \
        $ADV_LOG_ARGS \
        > "$log_file" 2>&1 &

    log "Server PID: $!"
}

wait_for_server() {
    local log_file="$1"
    local timeout=300  # 5 minutes max
    local elapsed=0

    log "Waiting for server to be ready (timeout ${timeout}s)..."
    while [ $elapsed -lt $timeout ]; do
        if grep -q "Running on all addresses" "$log_file" 2>/dev/null; then
            log "Server ready! (${elapsed}s)"
            sleep 2  # extra grace period
            return 0
        fi
        if ! pgrep -f "python benchmark/server.py" >/dev/null 2>&1; then
            log "ERROR: Server process died. Check $log_file"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "ERROR: Server did not start within ${timeout}s"
    return 1
}

run_benchmark() {
    local result_file="$1"
    log "Sending benchmark request..."
    local http_code
    http_code=$(curl -s -o "$result_file" -w "%{http_code}" \
        -X POST "http://localhost:${SERVER_PORT}/run_once" \
        -H "Content-Type: application/json" \
        -d "$BENCH_PAYLOAD" \
        --max-time 1800)

    if [ "$http_code" = "200" ]; then
        log "Benchmark completed (HTTP $http_code)"
        return 0
    elif [ "$http_code" = "000" ]; then
        log "ERROR: Benchmark timed out (1800s)"
        echo "error: curl timeout after 1800s" > "$result_file"
        return 1
    else
        log "ERROR: Benchmark returned HTTP $http_code"
        return 1
    fi
}

# ── Main ─────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"
log "Starting 8 experiments across 4 profiles × 2 logging modes"
log "Results dir: $RESULTS_DIR"

EXP_NUM=0
TOTAL=8

for profile_entry in "${PROFILES[@]}"; do
    IFS=: read -r profile_file label <<< "$profile_entry"
    gate_profile="${REPO_DIR}/gating_profiles/${profile_file}"

    if [ ! -f "$gate_profile" ]; then
        log "WARNING: Profile not found: $gate_profile — skipping"
        continue
    fi

    for logging in off on; do
        EXP_NUM=$((EXP_NUM + 1))
        exp_name="${label}_logging_${logging}"
        exp_dir="${RESULTS_DIR}/${exp_name}"
        mkdir -p "$exp_dir"

        log "════════════════════════════════════════════════"
        log "Experiment ${EXP_NUM}/${TOTAL}: ${exp_name}"
        log "════════════════════════════════════════════════"

        kill_server

        launch_server "$gate_profile" "$logging" "${exp_dir}/advanced_logs" "${exp_dir}/server.log"

        if ! wait_for_server "${exp_dir}/server.log"; then
            log "FAILED: ${exp_name} — server did not start. Skipping."
            echo '{"error": "server_start_failed"}' > "${exp_dir}/benchmark_result.json"
            continue
        fi

        if ! run_benchmark "${exp_dir}/benchmark_result.json"; then
            log "FAILED: ${exp_name} — benchmark request failed."
        fi

        # Give logging a moment to flush
        sleep 5

        log "Experiment ${exp_name} done."
    done
done

kill_server

log "════════════════════════════════════════════════"
log "All ${TOTAL} experiments complete."
log "Results in: ${RESULTS_DIR}"
log "════════════════════════════════════════════════"
