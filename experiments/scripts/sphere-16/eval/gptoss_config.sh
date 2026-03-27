#!/usr/bin/bash
# gptoss_config.sh — gptoss_120b model config (36 layers, 128 experts, top-4, bf16)

EVAL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EVAL_CONFIG_DIR/config.sh"

MODEL_NAME="gptoss_120b"
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"
NUM_SHARED_EXPERTS=0
