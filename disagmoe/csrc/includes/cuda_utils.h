#pragma once

#include "cuda_runtime.h"
#include "nccl.h"
#include "nvtx3/nvtx3.hpp"

#include <assert.h>
#include <cstdlib>
#include <cstdio>


#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


inline uintptr_t alloc_cuda_tensor(int count, int device_id) {
    assert(count > 0);
    // FIXME(hogura|20241001): replace float with half float
    void* data;
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(device_id));
    #endif
    CUDACHECK(cudaMalloc(&data, count * sizeof(short)));
    return (uintptr_t) (data);
}

inline uintptr_t alloc_copy_tensor(uintptr_t buf, int size) {
    void* data;
    CUDACHECK(cudaMalloc(&data, size));
    CUDACHECK(cudaMemcpy(data, (void*) buf, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    return (uintptr_t) data;
}

using tx_range = nvtx3::scoped_range;

#define AUTO_TX_RANGE tx_range __{__FUNCTION__}

// TODO(hogura|20241001): add allocAsync