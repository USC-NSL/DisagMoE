#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include "per_token_group_quant_8bit.cuh"

TORCH_LIBRARY(quant_fp8, m) {
  m.def(
      "sgl_per_token_group_quant_8bit(Tensor input, Tensor output_q, Tensor output_s, int group_size, "
      "float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
}

TORCH_LIBRARY_IMPL(quant_fp8, CUDA, m) {
  m.impl(
      "sgl_per_token_group_quant_8bit",
      TORCH_FN(sgl_per_token_group_quant_8bit));
}


