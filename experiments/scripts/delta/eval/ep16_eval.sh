#!/usr/bin/bash
# ep16_eval.sh — AsyncMoE EP16 final evaluation, NCSA Delta
#
# Usage:
#   source experiments/scripts/delta/env.sh     # conda amoe + LD_LIBRARY_PATH
#   bash experiments/scripts/delta/eval/ep16_eval.sh [RESULTS_DIR]
#
#   RESULTS_DIR  required; a parent directory that holds one sub-dir per run.
#                Example: /scratch/myrun/results
#
# Run directory naming: <RESULTS_DIR>/<system>-<dataset>/
#   e.g.  asyncmoe-sharegpt_regular/
#         asyncmoe-legal_court_balanced/
#
# Prerequisites:
#   - SLURM allocation active (4 nodes × 4 A100-SXM4-40GB = 16 GPUs)
#   - env.sh sourced in the current shell
#   - Gate profile parquets in place (see EXPERIMENT MATRIX below)

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load fixed config and function libraries ──────────────────────────────────
source "$EVAL_DIR/config.sh"
source "$EVAL_DIR/helpers/ray.sh"
source "$EVAL_DIR/helpers/server.sh"
source "$EVAL_DIR/helpers/benchmark.sh"

# ── Results directory (overridable via $1) ────────────────────────────────────
RESULTS_DIR="${1:?ERROR: RESULTS_DIR is required as the first argument (e.g. /path/to/results)}"

# ── Experiment matrix ─────────────────────────────────────────────────────────
# Format: "absolute_path_to_parquet:dataset_label"
# Run directories will be named: ${SYSTEM_NAME}-${dataset_label}
#
# Gate profiles — FILL IN paths before running.
# Regular profiles: captured from real inference traces.
# Balanced profiles: pre-generated and placed in gating_profiles/balanced_output/.

EXPERIMENTS=(
    "${GATING_DIR}/gating_gptoss120b_sharegpt_200.parquet:sharegpt_regular"                        # TODO: verify
    "${GATING_DIR}/balanced_output/balanced_gptoss120b_sharegpt_200.parquet:sharegpt_balanced"     # TODO: verify
    "${GATING_DIR}/gating_legal_court_opinions_200.parquet:legal_court_regular"                    # TODO: verify
    "${GATING_DIR}/balanced_output/balanced_legal_court_opinions_200.parquet:legal_court_balanced" # TODO: verify
)

MAX_RETRIES=3
MEM_FRAC_STEP=0.02   # how much to reduce MEM_FRAC on each OOM retry

# ─────────────────────────────────────────────────────────────────────────────
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [main] $*"; }

mkdir -p "$RESULTS_DIR"
log "EP16 evaluation starting"
log "  System      : $SYSTEM_NAME"
log "  Results dir : $RESULTS_DIR"
log "  Experiments : ${#EXPERIMENTS[@]}, up to $MAX_RETRIES retries each"
log "  Initial MEM_FRAC: $MEM_FRAC"

archive_attempt_artifacts() {
    local run_dir="$1"
    local attempt="$2"
    local archive_dir="$run_dir/attempt${attempt}"
    local moved=0

    mkdir -p "$archive_dir"
    for artifact in server.log server_cmd.sh bench_cmd.sh result.json; do
        if [ -e "$run_dir/$artifact" ]; then
            mv "$run_dir/$artifact" "$archive_dir/$artifact"
            moved=1
        fi
    done

    if [ "$moved" -eq 0 ]; then
        rmdir "$archive_dir" 2>/dev/null || true
    else
        log "Archived failed attempt $attempt artifacts to: $archive_dir"
    fi
}

WORKER_PIDS=()   # managed by lib/ray.sh
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
            # Server failed — check for OOM and back off memory fraction
            if is_oom "$server_log"; then
                new_frac=$(awk "BEGIN {printf \"%.2f\", $MEM_FRAC - $MEM_FRAC_STEP}")
                log "OOM detected — reducing MEM_FRAC: $MEM_FRAC -> $new_frac"
                MEM_FRAC="$new_frac"
            else
                log "Server failed (non-OOM). See: $server_log"
            fi
        fi

        kill_server
        archive_attempt_artifacts "$run_dir" "$attempt"
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
