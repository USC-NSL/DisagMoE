#!/usr/bin/bash
# =============================================================================
# Test shared experts: 1 node, 2 GPUs, 8 layers, FP8 attn + FP8 experts
# Usage:  conda activate disag12 && bash experiments/scripts/test_shared_expert.sh
#         conda activate disag12 && bash experiments/scripts/test_shared_expert.sh 4
# =============================================================================
set -euo pipefail

N_NODE=1
N_GPU_PER_NODE=2
WORLD_SIZE=$((N_NODE * N_GPU_PER_NODE))
SERVER_PORT=6699
NUM_SHARED_EXPERTS=${1:-2}

echo "============================================"
echo " Shared Expert Test (${NUM_SHARED_EXPERTS} shared experts)"
echo " FP8 attn + FP8 experts, 2 GPUs, 8 layers"
echo "============================================"

cleanup() {
    echo ""
    echo ">>> Cleaning up..."
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    ray stop --force 2>/dev/null || true
    echo ">>> Done."
}
trap cleanup EXIT

ray stop --force 2>/dev/null || true
sleep 2

echo ">>> Starting DisagMoE server..."
cd "$(dirname "$0")/../.."

python benchmark/server.py \
    -N $N_NODE \
    -g $N_GPU_PER_NODE \
    -u 0.98 \
    --model gptoss_120b \
    --num-layers 8 \
    --attn-qkv-quant fp8 \
    --moe-linear-quant fp8 \
    --num-shared-experts "$NUM_SHARED_EXPERTS" \
    --max-batch-size-attn 64 \
    --max-attn-graph-bsz 64 \
    --max-batch-size-expert 128 \
    --max-pending-sends 4 \
    --block-size 16 \
    --placement colocate \
    --dp-size $WORLD_SIZE \
    --ep-size $WORLD_SIZE \
    --transport zmq \
    --less-than-sm90 \
    --cuda-graph-attn \
    --cuda-graph-expert \
    --unified-scheduler-type defrag \
    --defrag-weight-decay 0.8 \
    --defrag-lookahead-steps 4 \
    --defrag-lookback-steps 4 \
    --analyze-throughput \
    --trace &

SERVER_PID=$!

echo ">>> Waiting for server on port ${SERVER_PORT}..."
MAX_WAIT=180
ELAPSED=0
while ! curl -s http://localhost:${SERVER_PORT}/ >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died during startup."
        wait "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s."
        exit 1
    fi
done
echo ">>> Server is up (took ~${ELAPSED}s)."

echo ""
echo ">>> Running benchmark: rate=10 req/s, 10s ..."
RESPONSE=$(curl -s -X POST http://localhost:${SERVER_PORT}/run_once \
    -H "Content-Type: application/json" \
    -d '{"rate": 10, "time": 10, "distribution": "poisson", "min_input_len": 30, "max_input_len": 70, "min_output_len": 40, "max_output_len": 80}')

echo ""
echo "============================================"
echo " Server Response"
echo "============================================"
echo "$RESPONSE"
echo ""

if echo "$RESPONSE" | grep -qi "success\|executed"; then
    echo ">>> TEST PASSED: Shared experts working end-to-end."
else
    echo ">>> TEST RESULT: Check response above for details."
fi
