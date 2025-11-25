#pragma once

#include <torch/extension.h>

// Per-token group FP8 quantization kernel (ported from disagmoe/ops/fp8_quantizer)
void sgl_per_token_group_quant_8bit(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0);



