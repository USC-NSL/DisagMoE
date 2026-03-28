#!/usr/bin/bash
# glm45air_config.sh — glm45air_106b model config (45 layers, 128 experts, top-8, bf16, 1 shared expert/layer)

EVAL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EVAL_CONFIG_DIR/config.sh"

MODEL_NAME="glm45air_106b"
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"
NUM_SHARED_EXPERTS=1
SHARED_EXPERT_INTERMEDIATE_SIZE=1408

