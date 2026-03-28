#!/usr/bin/bash
# glm45air_eval.sh — AsyncMoE EP16 glm45air evaluation, NCSA Delta
#
# Usage:
#   source experiments/scripts/delta/env.sh     # conda amoe + LD_LIBRARY_PATH
#   bash experiments/scripts/delta/eval/glm45air_eval.sh <RESULTS_DIR> [OPTIONS]
#
#   RESULTS_DIR  required; a parent directory that holds one sub-dir per run.
#                Example: /scratch/myrun/results
#
#   Options:
#     --list          Print numbered experiment list and exit
#     --only FILTER   Run only experiments matching FILTER (comma-separated
#                     indices or name substrings). Examples:
#                       --only 1,3              # by index
#                       --only sharegpt         # all sharegpt experiments
#                       --only sharegpt_regular # single experiment by exact label
#
# Run directory naming: <RESULTS_DIR>/<system>-<dataset>/
#   e.g.  asyncmoe-sharegpt_regular/
#         asyncmoe-gsm8k_balanced/
#
# Prerequisites:
#   - SLURM allocation active (4 nodes × 4 A100-SXM4-40GB = 16 GPUs)
#   - env.sh sourced in the current shell
#   - Gate profile parquets in place (see EXPERIMENT MATRIX below)

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$EVAL_DIR/glm45air_config.sh"
source "$EVAL_DIR/helpers/ray.sh"
source "$EVAL_DIR/helpers/server.sh"
source "$EVAL_DIR/helpers/benchmark.sh"

ONLY_FILTER=""
LIST_ONLY=0
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY_FILTER="${2:?ERROR: --only requires a comma-separated list}"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        -*) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$RESULTS_DIR" ]]; then RESULTS_DIR="$1"; shift
            else echo "ERROR: Unexpected argument: $1" >&2; exit 1; fi
            ;;
    esac
done

if [[ "$LIST_ONLY" -eq 0 ]] && [[ -z "$RESULTS_DIR" ]]; then
    echo "ERROR: RESULTS_DIR is required (e.g. /scratch/myrun/results)" >&2
    echo "Usage: $0 <RESULTS_DIR> [--list] [--only FILTER]" >&2
    exit 1
fi

GLM_GATING_DIR="$GATING_DIR/glm45air_gating_profiles"

EXPERIMENTS=(
    "${GLM_GATING_DIR}/gating_glm45air_sharegpt_200.parquet:sharegpt_regular"
    "${GLM_GATING_DIR}/balanced_output/balanced_glm45air_sharegpt_200.parquet:sharegpt_balanced"
    "${GLM_GATING_DIR}/gating_glm45air_gsm8k_200.parquet:gsm8k_regular"
    "${GLM_GATING_DIR}/balanced_output/balanced_glm45air_gsm8k_200.parquet:gsm8k_balanced"
)

MAX_RETRIES=3
MEM_FRAC_STEP=0.02

# ─────────────────────────────────────────────────────────────────────────────
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [main] $*"; }

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

# ── Experiment filter ─────────────────────────────────────────────────────────
should_run_experiment() {
    local idx="$1" label="$2"
    [[ -z "$ONLY_FILTER" ]] && return 0
    IFS=',' read -ra FILTERS <<< "$ONLY_FILTER"
    for f in "${FILTERS[@]}"; do
        f="${f#"${f%%[![:space:]]*}"}"
        f="${f%"${f##*[![:space:]]}"}"
        if [[ "$f" =~ ^[0-9]+$ ]]; then
            [[ "$f" -eq "$idx" ]] && return 0
        else
            [[ "$label" == *"$f"* ]] && return 0
        fi
    done
    return 1
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    echo "Available experiments:"
    _i=0
    for exp_entry in "${EXPERIMENTS[@]}"; do
        IFS=: read -r _gp _ds <<< "$exp_entry"
        _i=$((_i + 1))
        printf "  %2d. %s-%s\n" "$_i" "$SYSTEM_NAME" "$_ds"
    done
    exit 0
fi

WORKER_PIDS=()   # managed by helpers/ray.sh

mkdir -p "$RESULTS_DIR"
log "EP16 evaluation starting"
log "  System      : $SYSTEM_NAME"
log "  Model       : $MODEL_NAME (shared_experts=$NUM_SHARED_EXPERTS)"
log "  Results dir : $RESULTS_DIR"
log "  Cluster     : ${N_NODE} nodes × ${N_GPU_PER_NODE} GPUs (${WORLD_SIZE} total)"
log "  Experiments : ${#EXPERIMENTS[@]}, up to $MAX_RETRIES retries each"
log "  Initial MEM_FRAC: $MEM_FRAC"
[[ -n "$ONLY_FILTER" ]] && log "  Filter      : --only $ONLY_FILTER"

EXP_NUM=0
TOTAL=${#EXPERIMENTS[@]}

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=: read -r gate_profile dataset <<< "$exp_entry"
    EXP_NUM=$((EXP_NUM + 1))

    run_name="${SYSTEM_NAME}-${dataset}"

    if ! should_run_experiment "$EXP_NUM" "$run_name"; then
        log "[$EXP_NUM/$TOTAL] SKIP (--only filter): $run_name"
        continue
    fi

    run_dir="$RESULTS_DIR/$run_name"
    mkdir -p "$run_dir"

    if [[ "$dataset" == gsm8k* ]]; then
        BENCH_DATASET_PATH="${BENCH_DATASET_PATH:-$REPO_DIR/datasets/gsm8k_lengths.npy}"
    else
        BENCH_DATASET_PATH="${BENCH_DATASET_PATH:-$REPO_DIR/datasets/sharegpt_lengths.npy}"
    fi

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
                if is_oom "$server_log"; then
                    new_frac=$(awk "BEGIN {printf \"%.2f\", $MEM_FRAC - $MEM_FRAC_STEP}")
                    log "OOM detected during benchmark — reducing MEM_FRAC: $MEM_FRAC -> $new_frac"
                    MEM_FRAC="$new_frac"
                fi
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
        archive_attempt_artifacts "$run_dir" "$attempt"
        sleep 10
    done

    if [ "$SUCCESS" -eq 0 ]; then
        log "FAILED: $run_name — all $MAX_RETRIES attempts unsuccessful."
    else
        log "SUCCESS: $run_name"
    fi
done

kill_server
stop_ray

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All $TOTAL experiments done. Results in: $RESULTS_DIR"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
