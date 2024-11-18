#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

#include "cuda_utils.h"

torch::Tensor permute_tokens_cuda(torch::Tensor tokens, torch::Tensor mappings);

void gather_tokens_cuda(torch::Tensor dest, uintptr_t *src_ptr, int num_tokens, int hidden_size, cudaStream_t stream);