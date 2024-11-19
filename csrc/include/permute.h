#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

#include "cuda_utils.h"

torch::Tensor permute_tokens_cuda(torch::Tensor tokens, torch::Tensor mappings);

void gather_tokens_cuda(torch::Tensor dest, uintptr_t *src_ptr, int num_tokens, int hidden_size, cudaStream_t stream);

// a debug kernel
void add_one_cuda(float *d_out, float *d_in, int num_tokens, cudaStream_t stream);