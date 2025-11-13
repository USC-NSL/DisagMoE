#pragma once

#ifndef TENSOR_UTILS_H_
#define TENSOR_UTILS_H_

#include <torch/torch.h>
#include <torch/extension.h>
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

inline void rebind_1d_tensor(
    torch::Tensor& out,
    const torch::Tensor& base,
    int64_t offset_elems,
    int64_t length_elems
) {
  TORCH_CHECK(base.is_contiguous(), "base must be contiguous");
  out.set_(base.storage(),
           base.storage_offset() + offset_elems,
           {length_elems},
           {1});
}

inline void rebind_2d_tensor(
    torch::Tensor& out,
    const torch::Tensor& base,
    int64_t offset_elems,
    int64_t rows,
    int64_t cols,
    int64_t row_stride,
    int64_t col_stride = 1
) {
  // Interprets a 1D base buffer as a 2D view with custom stride
  out.set_(base.storage(),
           base.storage_offset() + offset_elems,
           {rows, cols},
           {row_stride, col_stride});
}

#endif