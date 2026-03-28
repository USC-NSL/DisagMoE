#!/usr/bin/bash
# lib/ray.sh — Ray cluster management helpers for Delta
# Source this file; do not execute directly.
#
# Requires (from config.sh):  MINICONDA, CONDA_ENV, REPO_DIR
# Requires (from environment): SLURM_JOB_ID, SLURM_JOB_NODELIST
#
# Exports/mutates:
#   WORKER_PIDS — array of background srun PIDs; caller must declare it first:
#       WORKER_PIDS=()

log_ray() { echo "$(date '+%Y-%m-%d %H:%M:%S') [ray] $*"; }

ray_bin() {
    printf '%s/envs/%s/bin/ray' "$MINICONDA" "$CONDA_ENV"
}

# restart_ray
#   Kills any existing server + srun worker steps, then brings up a fresh
#   Ray cluster (head on current node, workers on remaining SLURM nodes).
#   Blocks until workers have had 60 s to join, then prints ray status.
restart_ray() {
    log_ray "=== Restarting Ray cluster ==="

    # Kill previous server and srun worker steps
    _ray_kill_workers
    sleep 5

    # Validate SLURM context
    [ -z "${SLURM_JOB_ID:-}" ] && {
        echo "[ray] FATAL: SLURM_JOB_ID not set — must run inside a SLURM allocation"
        return 1
    }

    # Resolve node list
    ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
    WORKER_NODES=("${ALL_NODES[@]:1}")
    # Management-network IP for Ray control plane (not hsn0 which is data-only)
    # Allow HEAD_IP to be pre-set via env var (fallback: socket trick, requires internet)
    if [ -z "${HEAD_IP:-}" ]; then
        HEAD_IP=$(hostname -I | awk '{print $1}')
    fi

    # Stop ray on all nodes
    log_ray "Stopping ray on all nodes..."
    "$(ray_bin)" stop 2>/dev/null || true
    for node in "${WORKER_NODES[@]}"; do
        srun --jobid="$SLURM_JOB_ID" --nodelist="$node" --overlap bash -c \
            "source $MINICONDA/etc/profile.d/conda.sh && \
             conda activate $CONDA_ENV && ray stop 2>/dev/null || true" &
    done
    wait
    sleep 5

    # Start ray head on current node
    log_ray "Starting ray head (IP: $HEAD_IP)..."
    export RAY_TMPDIR=/tmp/ray
    "$(ray_bin)" start --head --port=6379 \
        --min-worker-port=30000 --max-worker-port=39999 \
        --disable-usage-stats

    # Start ray worker on each other node; keep srun step alive with sleep infinity
    log_ray "Starting ray workers on: ${WORKER_NODES[*]}"
    for node in "${WORKER_NODES[@]}"; do
        srun --jobid="$SLURM_JOB_ID" --nodelist="$node" --overlap bash -c \
            "source $MINICONDA/etc/profile.d/conda.sh && \
             conda activate $CONDA_ENV && \
             source $REPO_DIR/experiments/scripts/delta/env.sh && \
             export RAY_TMPDIR=/tmp/ray && \
             ray start --address=${HEAD_IP}:6379 --disable-usage-stats && \
             sleep infinity" &
        WORKER_PIDS+=($!)
    done

    log_ray "Waiting 60s for workers to join..."
    sleep 60
    log_ray "Ray status:"
    "$(ray_bin)" status
}

# stop_ray
#   Gracefully stops ray and kills all tracked srun worker steps.
stop_ray() {
    log_ray "=== Stopping Ray cluster ==="
    _ray_kill_workers
    "$(ray_bin)" stop 2>/dev/null || true
    log_ray "Ray stopped."
}

# _ray_kill_workers  (internal)
#   Kills tracked background srun worker PIDs and clears WORKER_PIDS.
_ray_kill_workers() {
    for pid in "${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    WORKER_PIDS=()
}
