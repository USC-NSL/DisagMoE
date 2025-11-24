#pragma once
#ifndef CUDA_GRAPH_H
#define CUDA_GRAPH_H

#include <torch/torch.h>
#include <torch/extension.h>

#include "cuda_utils.h"

void cuda_graph_preprocess_fused(
    torch::Tensor hidden,
    torch::Tensor positions,
    torch::Tensor block_tables,
    torch::Tensor slot_mapping,
    torch::Tensor seq_lens,
    torch::Tensor context_lens,
    torch::Tensor seq_start_loc,

    torch::Tensor out_hidden,
    torch::Tensor out_positions,
    torch::Tensor out_block_tables,
    torch::Tensor out_slot_mapping,
    torch::Tensor out_seq_lens,
    torch::Tensor out_context_lens,
    torch::Tensor out_seq_start_loc,

    int tokens_per_block,
    uintptr_t raw_cuda_stream
);

#endif // CUDA_GRAPH_H