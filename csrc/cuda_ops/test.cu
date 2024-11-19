#include <cuda_runtime.h>
#include <cuda.h>

#include "tests.h"

__global__ void add_one_kernel(float *d_out, float *d_in) {
    int token_id = blockIdx.x;
    d_out[token_id] = d_in[token_id] + 1;
}

void add_one_cuda(float *d_out, float *d_in, int num_tokens, cudaStream_t stream) {
    add_one_kernel<<<num_tokens, 1, 0, stream>>>(d_out, d_in);
}