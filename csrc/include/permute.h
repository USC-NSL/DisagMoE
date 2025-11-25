#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

#include "cuda_utils.h"

torch::Tensor permute_tokens_cuda_dispatch(torch::Tensor tokens, torch::Tensor mappings, int64_t raw_cuda_stream);

void gather_tokens_cuda_dispatch(torch::Tensor dest, int64_t src_ptr, int64_t num_tokens, int64_t hidden_size, int64_t raw_cuda_stream);