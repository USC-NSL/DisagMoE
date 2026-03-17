#!/usr/bin/bash
set -euo pipefail

# Start CUDA MPS with 50% active thread percentage on the last 4 nodes
# (sgpu6, sgpu7, sgpu8, sgpu9) to simulate heterogeneous compute.

THROTTLED_NODES=(sgpu6 sgpu7 sgpu8 sgpu9)
THREAD_PCT=50

echo "Setting up MPS throttling (${THREAD_PCT}% threads) on: ${THROTTLED_NODES[*]}"

for node in "${THROTTLED_NODES[@]}"; do
    echo "=== $node ==="
    ssh "$node" bash -s "$THREAD_PCT" <<'REMOTE'
        PCT="$1"
        # Stop any existing MPS
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        sleep 1

        # Set the default active thread percentage and start daemon
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PCT"
        nvidia-cuda-mps-control -d
        sleep 1

        # Verify
        RESULT=$(echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control 2>/dev/null || echo "unknown")
        echo "  MPS daemon started, default active thread percentage: $RESULT"
REMOTE
done

echo "MPS throttling setup complete."
