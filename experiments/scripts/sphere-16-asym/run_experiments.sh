#!/usr/bin/bash
set -euo pipefail

# Sphere-16 Asymmetric Experiment Runner
# Compares 1:1 vs 2:1 DP attention weights with MPS-throttled heterogeneous cluster.
#
# Cluster: 8 nodes × 2 L40S GPUs = 16 GPUs
#   - Nodes sgpu0, sgpu2, sgpu3, sgpu4 (8 GPUs): full compute
#   - Nodes sgpu6, sgpu7, sgpu8, sgpu9 (8 GPUs): 50% compute via CUDA MPS
#
# Expert allocation: 128 total
#   - Full-compute GPUs: 10-11 experts each (84 total)
#   - Throttled GPUs: 5-6 experts each (44 total)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$HOME/DisagMoE"
RESULTS_DIR="${SCRIPT_DIR}/results"
CONDA_DIR="$HOME/miniconda3"

ALL_NODES="sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9"

# Server config (matches sphere-16 except asymmetric + no trace)
N_NODE=8
N_GPU_PER_NODE=2
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))
MODEL_NAME="gptoss_120b"
GATE_PROFILE="${REPO_DIR}/gating_profiles/gating_math_gsm8k_200.parquet"
SERVER_PORT=6699
MAX_PENDING_SENDS=16

# Benchmark payload — doubled input length range vs sphere-16 (128-256 instead of 64-128)
BENCH_PAYLOAD='{
    "rate": 10,
    "time": 5,
    "distribution": "poisson",
    "min_input_len": 128,
    "max_input_len": 256,
    "min_output_len": 256,
    "max_output_len": 512
}'

# Experiments: (config_file, label)
EXPERIMENTS=(
    "asym_expert_alloc_equal_weights.json:equal_1to1"
    "asym_expert_alloc_2to1_weights.json:weighted_2to1"
)

log() { echo "$(date '+%H:%M:%S') [EXP] $*"; }

kill_server() {
    log "Killing server..."
    pkill -f "python benchmark/server.py" 2>/dev/null || true
    sleep 5
    if pgrep -f "python benchmark/server.py" >/dev/null 2>&1; then
        pkill -9 -f "python benchmark/server.py" 2>/dev/null || true
        sleep 3
    fi
    # Give GPUs time to clean up after killing
    sleep 10
}

launch_server() {
    local alloc_file="$1"
    local log_file="$2"

    cd "$REPO_DIR"
    nohup env NCCL_RUNTIME_CONNECT="${NCCL_RUNTIME_CONNECT:-0}" python benchmark/server.py \
        -N $N_NODE \
        -g $N_GPU_PER_NODE \
        -u 0.98 \
        --model $MODEL_NAME \
        --attn-qkv-quant none \
        --moe-linear-quant none \
        --max-batch-size-attn 256 \
        --max-attn-graph-bsz 256 \
        --max-pending-sends $MAX_PENDING_SENDS \
        --max-batch-size-exp 512 \
        --block-size 16 \
        --placement colocate \
        --dp-size $WORLD_SIZE \
        --ep-size $WORLD_SIZE \
        --transport zmq \
        --host-ifname ens1f1np1 \
        --nccl-ib-hca mlx5_1 \
        --nccl-ib-gid-index 3 \
        --unified-scheduler-type defrag \
        --defrag-weight-decay 0.8 \
        --defrag-lookahead-steps 4 \
        --defrag-lookback-steps 4 \
        --less-than-sm90 \
        --cuda-graph-attn \
        --cuda-graph-expert \
        --file "${RESULTS_DIR}/benchmark.csv" \
        --analyze-throughput \
        --gate-profile-file "$GATE_PROFILE" \
        --expert-allocation-path "$alloc_file" \
        > "$log_file" 2>&1 &

    log "Server PID: $!"
}

wait_for_server() {
    local log_file="$1"
    local timeout=600
    local elapsed=0

    log "Waiting for server (timeout ${timeout}s)..."
    while [ $elapsed -lt $timeout ]; do
        if grep -q "Running on all addresses" "$log_file" 2>/dev/null; then
            log "Server ready (${elapsed}s)"
            sleep 2
            return 0
        fi
        if ! pgrep -f "python benchmark/server.py" >/dev/null 2>&1; then
            log "ERROR: Server died. Check $log_file"
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    log "ERROR: Timeout"
    return 1
}

run_benchmark() {
    local result_file="$1"
    log "Sending benchmark..."
    local http_code
    http_code=$(curl -s -o "$result_file" -w "%{http_code}" \
        -X POST "http://localhost:${SERVER_PORT}/run_once" \
        -H "Content-Type: application/json" \
        -d "$BENCH_PAYLOAD" \
        --max-time 1800)

    if [ "$http_code" = "200" ]; then
        log "Benchmark completed (HTTP $http_code)"
        cat "$result_file"
        return 0
    else
        log "ERROR: HTTP $http_code"
        return 1
    fi
}

# ── Main ─────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"
log "Starting sphere-16-asym experiments (2 configs)"

EXP_NUM=0
for entry in "${EXPERIMENTS[@]}"; do
    IFS=: read -r config_file label <<< "$entry"
    alloc_path="${SCRIPT_DIR}/${config_file}"
    EXP_NUM=$((EXP_NUM + 1))

    exp_dir="${RESULTS_DIR}/${label}"
    mkdir -p "$exp_dir"

    log "════════════════════════════════════════════════"
    log "Experiment ${EXP_NUM}/2: ${label}"
    log "  Config: ${config_file}"
    log "════════════════════════════════════════════════"

    kill_server

    launch_server "$alloc_path" "${exp_dir}/server.log"

    if ! wait_for_server "${exp_dir}/server.log"; then
        log "FAILED: ${label} — server did not start."
        echo '{"error": "server_start_failed"}' > "${exp_dir}/benchmark_result.txt"
        continue
    fi

    if ! run_benchmark "${exp_dir}/benchmark_result.txt"; then
        log "FAILED: ${label} — benchmark failed."
    fi

    sleep 5
    log "Experiment ${label} done."
done

kill_server

log "════════════════════════════════════════════════"
log "All experiments complete. Results in: ${RESULTS_DIR}"
log "════════════════════════════════════════════════"
