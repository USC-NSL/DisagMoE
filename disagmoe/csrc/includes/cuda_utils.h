#pragma once

#include "cuda_runtime.h"
#include "nccl.h"

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
    printf("device: %d %d\n", device_id, count);
    // FIXME(hogura|20241001): replace float with half float
    void* data;
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(device_id));
    #endif
    CUDACHECK(cudaMalloc(&data, count * sizeof(short)));
    printf("allocated cuda addr: %u, count: %d\n", (uintptr_t) data, count);
    return (uintptr_t) (data);
}

// TODO(hogura|20241001): add allocAsync