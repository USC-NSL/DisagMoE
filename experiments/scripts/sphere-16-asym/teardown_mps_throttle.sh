#!/usr/bin/bash
set -euo pipefail

# Stop CUDA MPS on the throttled nodes.

THROTTLED_NODES=(sgpu6 sgpu7 sgpu8 sgpu9)

echo "Tearing down MPS on: ${THROTTLED_NODES[*]}"

for node in "${THROTTLED_NODES[@]}"; do
    echo "=== $node ==="
    ssh "$node" bash -s <<'REMOTE'
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        sleep 1
        # Verify no MPS daemon running
        if pgrep -f nvidia-cuda-mps 2>/dev/null; then
            echo "  WARNING: MPS processes still running"
        else
            echo "  MPS stopped."
        fi
REMOTE
done

echo "MPS teardown complete."
