#pragma once

#ifndef TENSOR_UTILS_H_
#define TENSOR_UTILS_H_

#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <cassert>
#include <cstdint>
#include "constants.h"

#define GPU_PAGE_SIZE (1 << 16)

inline torch::Tensor get_cuda_aligned_tensor(
    int numel,
    torch::ScalarType dtype,
    int alignment = GPU_PAGE_SIZE
) {
    // ---- 1. Compute element size ----
    int elem_size = c10::elementSize(dtype);
    int size_bytes = numel * elem_size;

    // ---- 2. Overallocate (uint8 buffer) ----
    torch::Tensor buf = torch::empty(
        {size_bytes + alignment},
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA)
    );

    // ---- 3. Compute aligned pointer ----
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(buf.data_ptr<uint8_t>());
    uintptr_t aligned_addr = (base_addr + alignment - 1) & ~(alignment - 1);
    int offset = static_cast<int>(aligned_addr - base_addr);

    // ---- 4. Slice the buffer: buf[offset : offset + size_bytes] ----
    torch::Tensor aligned_buf = buf.slice(/*dim=*/0, offset, offset + size_bytes);

    return aligned_buf.view(dtype);
}
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