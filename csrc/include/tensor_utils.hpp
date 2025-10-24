#pragma once

#ifndef TENSOR_UTILS_H_
#define TENSOR_UTILS_H_

#include <torch/torch.h>
#include <vector>
#include <cassert>

inline std::vector<torch::Tensor> split_tensor_by_indice(const torch::Tensor &tensor, const std::vector<int> &indices) {
    std::vector<torch::Tensor> res{};
    ASSERT (tensor.size(0) == indices.back());

    for (size_t i = 0; i < indices.size() - 1; i ++) {
        int l = indices[i];
        int r = indices[i + 1];
        res.emplace_back(tensor.slice(0, l, r));
    }
    return res;
}

inline std::vector<torch::Tensor> split_tensor_by_size(const torch::Tensor &tensor, const std::vector<int> &sizes) {
    std::vector<int64_t> sizes64(sizes.begin(), sizes.end());
    return torch::split(tensor, sizes64, 0);
}

#endif