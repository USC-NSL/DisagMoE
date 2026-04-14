#!/usr/bin/bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate disag12
cd ~/DisagMoE

RESULTS=~/DisagMoE/experiments/sphere-16-investigate-glm/asyncmoe-glm45air-results
REPO_DIR=~/DisagMoE
mkdir -p "$RESULTS"

echo "=========================================="
echo "Starting AsyncMoE GLM investigation runs"
echo "Results: $RESULTS"
echo "=========================================="

# Experiment 1: sharegpt_regular (with advanced logging)
echo ""
echo "[1/4] sharegpt_regular (with advanced logging)"
SERVER_EXTRA_ARGS="--enable-advanced-logging --advanced-logging-dir $RESULTS/asyncmoe-sharegpt_regular/advanced_logs" \
BENCH_RATE=1000 \
bash experiments/scripts/sphere-16/eval/glm45air_eval.sh "$RESULTS" --only sharegpt_regular \
    2>&1 | tee "$RESULTS/eval_sharegpt_regular.log"

# Experiment 2: sharegpt_balanced
echo ""
echo "[2/4] sharegpt_balanced"
BENCH_RATE=1000 \
bash experiments/scripts/sphere-16/eval/glm45air_eval.sh "$RESULTS" --only sharegpt_balanced \
    2>&1 | tee "$RESULTS/eval_sharegpt_balanced.log"

# Experiment 3: gsm8k_regular
echo ""
echo "[3/4] gsm8k_regular"
BENCH_DATASET_PATH=$REPO_DIR/datasets/gsm8k_lengths.npy \
bash experiments/scripts/sphere-16/eval/glm45air_eval.sh "$RESULTS" --only gsm8k_regular \
    2>&1 | tee "$RESULTS/eval_gsm8k_regular.log"

# Experiment 4: gsm8k_balanced
echo ""
echo "[4/4] gsm8k_balanced"
BENCH_DATASET_PATH=$REPO_DIR/datasets/gsm8k_lengths.npy \
bash experiments/scripts/sphere-16/eval/glm45air_eval.sh "$RESULTS" --only gsm8k_balanced \
    2>&1 | tee "$RESULTS/eval_gsm8k_balanced.log"

echo ""
echo "=========================================="
echo "All 4 experiments done. Extracting metrics..."
echo "=========================================="

# Extract metrics
python experiments/evaluations/parse_detokenizer_logs.py \
    "$RESULTS" \
    --csv "$RESULTS/metrics.csv" \
    --window 15,45 2>&1 || echo "Metric extraction failed (may need manual window adjustment)"

# Process advanced logs from experiment 1
if [ -d "$RESULTS/asyncmoe-sharegpt_regular/advanced_logs" ]; then
    echo "Processing advanced logs..."
    python experiments/process_and_plot_advanced_logging.py \
        "$RESULTS/asyncmoe-sharegpt_regular/advanced_logs" \
        "$RESULTS/asyncmoe-sharegpt_regular/plots" 2>&1 || echo "Advanced log processing failed"
fi

echo ""
echo "All done. Check $RESULTS for outputs."
