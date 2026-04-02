#!/usr/bin/bash
# lib/server.sh — DisagMoE server lifecycle helpers for Sphere-16
# Source this file; do not execute directly.
#
# Requires (from config.sh):
#   REPO_DIR, SERVER_PORT, SERVER_READY_TIMEOUT
#   N_NODE, N_GPU_PER_NODE, MEM_FRAC
#   MODEL_NAME, ATTN_QKV_QUANT, MOE_LINEAR_QUANT
#   MAX_BATCH_SIZE_ATTN, MAX_BATCH_SIZE_EXP, MAX_PENDING_SENDS, BLOCK_SIZE
#   PLACEMENT, DP_SIZE, EP_SIZE, TRANSPORT
#   HOST_IFNAME, NCCL_IB_HCA, NCCL_IB_GID_INDEX
#   UNIFIED_SCHEDULER_TYPE, DEFRAG_WEIGHT_DECAY,
#   DEFRAG_LOOKAHEAD_STEPS, DEFRAG_LOOKBACK_STEPS
#   NUM_SHARED_EXPERTS (optional, default 0),
#   SHARED_EXPERT_INTERMEDIATE_SIZE (optional)

log_server() { echo "$(date '+%Y-%m-%d %H:%M:%S') [server] $*"; }

server_python() {
    printf '%s/envs/%s/bin/python' "$MINICONDA" "$CONDA_ENV"
}

# launch_server <gate_profile_path> <server_log_path> <cmd_file_path>
launch_server() {
    local gate_profile="$1"
    local server_log="$2"
    local cmd_file="$3"

    log_server "Launching server | profile: $(basename "$gate_profile") | mem_frac=$MEM_FRAC"
    mkdir -p "$(dirname "$server_log")"

    local cmd=(
        "$(server_python)" benchmark/server.py
        -N "$N_NODE"
        -g "$N_GPU_PER_NODE"
        -u "$MEM_FRAC"
        --model "$MODEL_NAME"
        --attn-qkv-quant "$ATTN_QKV_QUANT"
        --moe-linear-quant "$MOE_LINEAR_QUANT"
        --max-batch-size-attn "$MAX_BATCH_SIZE_ATTN"
        --max-attn-graph-bsz "$MAX_BATCH_SIZE_ATTN"
        --max-pending-sends "$MAX_PENDING_SENDS"
        --max-batch-size-expert "$MAX_BATCH_SIZE_EXP"
        --block-size "$BLOCK_SIZE"
        --placement "$PLACEMENT"
        --dp-size "$DP_SIZE"
        --ep-size "$EP_SIZE"
        --transport "$TRANSPORT"
        --host-ifname "$HOST_IFNAME"
        --nccl-ib-hca "$NCCL_IB_HCA"
        --nccl-ib-gid-index "$NCCL_IB_GID_INDEX"
        --unified-scheduler-type "$UNIFIED_SCHEDULER_TYPE"
        --defrag-weight-decay "$DEFRAG_WEIGHT_DECAY"
        --defrag-lookahead-steps "$DEFRAG_LOOKAHEAD_STEPS"
        --defrag-lookback-steps "$DEFRAG_LOOKBACK_STEPS"
        --less-than-sm90
        --cuda-graph-attn
        --cuda-graph-expert
        --analyze-throughput
        --trace
        --gate-profile-file "$gate_profile"
    )

    if [ -n "${ANALYZE_THROUGHPUT_WINDOW:-}" ]; then
        cmd+=(--analyze-throughput-window "$ANALYZE_THROUGHPUT_WINDOW")
    fi

    if [ "${NUM_SHARED_EXPERTS:-0}" -gt 0 ]; then
        cmd+=(--num-shared-experts "$NUM_SHARED_EXPERTS")
        if [ -n "${SHARED_EXPERT_INTERMEDIATE_SIZE:-}" ]; then
            cmd+=(--shared-expert-intermediate-size "$SHARED_EXPERT_INTERMEDIATE_SIZE")
        fi
    fi

    if [ -n "${SERVER_EXTRA_ARGS:-}" ]; then
        read -ra _extra <<< "$SERVER_EXTRA_ARGS"
        cmd+=("${_extra[@]}")
    fi

    {
        printf '# Server command\n'
        printf '# Generated: %s\n' "$(date)"
        printf '# mem_frac: %s\n\n' "$MEM_FRAC"
        printf 'cd %s\n' "$REPO_DIR"
        printf 'nohup env NCCL_RUNTIME_CONNECT=0'
        for arg in "${cmd[@]}"; do printf ' \\\n    %q' "$arg"; done
        printf ' \\\n    > %q 2>&1 &\n' "$server_log"
    } > "$cmd_file"

    cd "$REPO_DIR"
    nohup env NCCL_RUNTIME_CONNECT=0 "${cmd[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!
    log_server "Server PID: $SERVER_PID (command saved to $(basename "$cmd_file"))"
}

# wait_for_server <server_log_path>
wait_for_server() {
    local server_log="$1"
    local elapsed=0

    log_server "Waiting for server ready (timeout ${SERVER_READY_TIMEOUT}s)..."
    while [ "$elapsed" -lt "$SERVER_READY_TIMEOUT" ]; do
        if grep -q "Running on all addresses\|Running on http://0.0.0.0" \
                "$server_log" 2>/dev/null; then
            log_server "Server ready (${elapsed}s elapsed)."
            sleep 3
            return 0
        fi
        if ! pgrep -f "benchmark/server.py" >/dev/null 2>&1; then
            log_server "ERROR: Server process exited unexpectedly. See: $server_log"
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    log_server "ERROR: Server did not become ready within ${SERVER_READY_TIMEOUT}s."
    return 1
}

# is_oom <server_log_path>
is_oom() {
    local server_log="$1"
    grep -qi \
        "out of memory\|OutOfMemoryError\|CUDA error: out of memory\|cudaMalloc failed" \
        "$server_log" 2>/dev/null
}

# kill_server
kill_server() {
    log_server "Killing server..."
    pkill -f "benchmark/server.py" 2>/dev/null || true
    sleep 5
    pkill -9 -f "benchmark/server.py" 2>/dev/null || true
    log_server "Server killed."
}
