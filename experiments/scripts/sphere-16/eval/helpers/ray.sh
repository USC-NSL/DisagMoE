#!/usr/bin/bash
# lib/ray.sh — Ray cluster management helpers for Sphere-16
# Source this file; do not execute directly.
#
# Requires (from config.sh):  MINICONDA, CONDA_ENV, HEAD_IP, WORKER_NODES

log_ray() { echo "$(date '+%Y-%m-%d %H:%M:%S') [ray] $*"; }

ray_bin() {
    printf '%s/envs/%s/bin/ray' "$MINICONDA" "$CONDA_ENV"
}

# restart_ray
#   Kills any existing server + SSH ray workers, then brings up a fresh
#   Ray cluster (head on sgpu0, workers on remaining nodes via SSH).
restart_ray() {
    log_ray "=== Restarting Ray cluster ==="

    kill_server 2>/dev/null || true
    sleep 5

    # Stop ray on all nodes
    log_ray "Stopping ray on all nodes..."
    "$(ray_bin)" stop 2>/dev/null || true
    for node in "${WORKER_NODES[@]}"; do
        ssh "$node" "source $MINICONDA/etc/profile.d/conda.sh && \
            conda activate $CONDA_ENV && ray stop 2>/dev/null || true" &
    done
    wait
    sleep 5

    # Start ray head on current node
    log_ray "Starting ray head (IP: $HEAD_IP)..."
    "$(ray_bin)" start --head --node-ip-address="$HEAD_IP" --port=6379 \
        --dashboard-port=8265 \
        --min-worker-port=30000 --max-worker-port=39999 \
        --disable-usage-stats

    # Start ray worker on each other node via SSH
    log_ray "Starting ray workers on: ${WORKER_NODES[*]}"
    for node in "${WORKER_NODES[@]}"; do
        ssh "$node" "source $MINICONDA/etc/profile.d/conda.sh && \
            conda activate $CONDA_ENV && \
            ray start --address=${HEAD_IP}:6379 --disable-usage-stats" &
    done
    wait

    log_ray "Waiting 30s for workers to join..."
    sleep 30
    log_ray "Ray status:"
    "$(ray_bin)" status
}

# stop_ray
#   Gracefully stops ray on all nodes.
stop_ray() {
    log_ray "=== Stopping Ray cluster ==="
    for node in "${WORKER_NODES[@]}"; do
        ssh "$node" "source $MINICONDA/etc/profile.d/conda.sh && \
            conda activate $CONDA_ENV && ray stop 2>/dev/null || true" &
    done
    wait
    "$(ray_bin)" stop 2>/dev/null || true
    log_ray "Ray stopped."
}
