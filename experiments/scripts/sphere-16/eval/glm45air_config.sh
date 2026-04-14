#!/usr/bin/bash
# glm45air_config.sh — glm45air_106b model config (45 layers, 128 experts, top-8, bf16, 1 shared expert/layer)

EVAL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EVAL_CONFIG_DIR/config.sh"

MODEL_NAME="glm45air_106b"
ATTN_QKV_QUANT="none"
MOE_LINEAR_QUANT="none"
NUM_SHARED_EXPERTS=1
SHARED_EXPERT_INTERMEDIATE_SIZE=1408

# GLM-4.5-Air needs lower initial MEM_FRAC due to shared expert + top-8 routing memory overhead
# Hard override: config.sh sets 0.98, but GLM OOMs at runtime with that value
MEM_FRAC=${MEM_FRAC_OVERRIDE:-0.90}
