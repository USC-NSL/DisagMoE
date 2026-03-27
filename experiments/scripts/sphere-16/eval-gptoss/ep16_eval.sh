#!/usr/bin/bash
# ep16_eval.sh — AsyncMoE EP16 gptoss evaluation, Sphere-16
#
# Usage:
#   conda activate disag12
#   bash experiments/scripts/sphere-16/eval-gptoss/ep16_eval.sh <RESULTS_DIR>
#
#   RESULTS_DIR  required; a parent directory that holds one sub-dir per run.
#                Example: ~/results/ep16_gptoss
#
# Run directory naming: <RESULTS_DIR>/<system>-<dataset>/
#   e.g.  asyncmoe-sharegpt_regular/
#         asyncmoe-legal_court_balanced/
#
# Prerequisites:
#   - Run from sgpu0 with disag12 conda env active
#   - SSH access to all worker nodes (sgpu2-9)
#   - Gate profile parquets in place (see EXPERIMENT MATRIX below)

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load fixed config and function libraries ──────────────────────────────────
source "$EVAL_DIR/config.sh"
source "$EVAL_DIR/helpers/ray.sh"
source "$EVAL_DIR/helpers/server.sh"
source "$EVAL_DIR/helpers/benchmark.sh"

# ── Results directory (required as $1) ────────────────────────────────────────
RESULTS_DIR="${1:?ERROR: RESULTS_DIR is required as the first argument (e.g. ~/results/ep16_gptoss)}"

# ── Experiment matrix ─────────────────────────────────────────────────────────
# Format: "absolute_path_to_parquet:dataset_label"
# Run directories will be named: ${SYSTEM_NAME}-${dataset_label}
#
# Gate profiles — FILL IN paths before running.

EXPERIMENTS=(
    "${GATING_DIR}/gating_gptoss120b_sharegpt_200.parquet:sharegpt_regular"                        # TODO: verify
    "${GATING_DIR}/balanced_output/balanced_gptoss120b_sharegpt_200.parquet:sharegpt_balanced"     # TODO: verify
    "${GATING_DIR}/gating_legal_court_opinions_200.parquet:legal_court_regular"                    # TODO: verify
    "${GATING_DIR}/balanced_output/balanced_legal_court_opinions_200.parquet:legal_court_balanced" # TODO: verify
)

MAX_RETRIES=3
MEM_FRAC_STEP=0.02

# ─────────────────────────────────────────────────────────────────────────────
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [main] $*"; }

mkdir -p "$RESULTS_DIR"
log "EP16 evaluation starting"
log "  System      : $SYSTEM_NAME"
log "  Results dir : $RESULTS_DIR"
log "  Cluster     : ${N_NODE} nodes × ${N_GPU_PER_NODE} GPUs (${WORLD_SIZE} total)"
log "  Experiments : ${#EXPERIMENTS[@]}, up to $MAX_RETRIES retries each"
log "  Initial MEM_FRAC: $MEM_FRAC"

EXP_NUM=0
TOTAL=${#EXPERIMENTS[@]}

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=: read -r gate_profile dataset <<< "$exp_entry"
    EXP_NUM=$((EXP_NUM + 1))

    run_name="${SYSTEM_NAME}-${dataset}"
    run_dir="$RESULTS_DIR/$run_name"
    mkdir -p "$run_dir"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "[$EXP_NUM/$TOTAL] $run_name"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$gate_profile" ]; then
        log "SKIP: profile not found: $gate_profile"
        printf '{"error":"profile_not_found","path":"%s"}\n' "$gate_profile" \
            > "$run_dir/result.json"
        continue
    fi

    SUCCESS=0
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        log "Attempt $attempt/$MAX_RETRIES (MEM_FRAC=$MEM_FRAC)..."

        restart_ray || { log "Ray restart failed; aborting experiment."; break; }

        server_log="$run_dir/server.log"
        server_cmd="$run_dir/server_cmd.sh"
        launch_server "$gate_profile" "$server_log" "$server_cmd"

        if wait_for_server "$server_log"; then
            bench_result="$run_dir/result.json"
            bench_cmd="$run_dir/bench_cmd.sh"
            if run_benchmark "$bench_result" "$bench_cmd"; then
                SUCCESS=1
                break
            else
                log "Benchmark failed on attempt $attempt."
            fi
        else
            if is_oom "$server_log"; then
                new_frac=$(awk "BEGIN {printf \"%.2f\", $MEM_FRAC - $MEM_FRAC_STEP}")
                log "OOM detected — reducing MEM_FRAC: $MEM_FRAC -> $new_frac"
                MEM_FRAC="$new_frac"
            else
                log "Server failed (non-OOM). See: $server_log"
            fi
        fi

        kill_server
        sleep 10
    done

    if [ "$SUCCESS" -eq 0 ]; then
        log "FAILED: $run_name — all $MAX_RETRIES attempts unsuccessful."
    else
        log "SUCCESS: $run_name"
    fi
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill_server
stop_ray

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All $TOTAL experiments done. Results in: $RESULTS_DIR"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
