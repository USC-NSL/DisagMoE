#pragma once

#include <torch/extension.h>

void fused_copy_and_pad(
    torch::Tensor in_hiddens,
    torch::Tensor in_m_indices,
    torch::Tensor out_hiddens,
    torch::Tensor out_m_indices,
    int64_t bs, int64_t bucket_bs);

