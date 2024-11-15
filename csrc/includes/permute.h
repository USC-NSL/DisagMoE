#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor permute_tokens_cuda(torch::Tensor tokens, torch::Tensor mappings);