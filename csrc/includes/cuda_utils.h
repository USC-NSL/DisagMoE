#pragma once

#include "cuda_runtime.h"
#include "nccl.h"
#include "constants.h"

#include <execinfo.h>
#include <cstdlib>
#include <unistd.h>
#include <cstdio>


static void print_back_trace() {
    // void *array[16];
    // size_t size = backtrace(array, 16);
    // backtrace_symbols_fd(array, size, STDERR_FILENO);
}

#define CUDACHECK(cmd) do {                             \
    cudaError_t err = cmd;                              \
    if (err != cudaSuccess) {                           \
        printf("Failed: Cuda error %s:%d '%s'\n",       \
            __FILE__,__LINE__,cudaGetErrorString(err)); \
        print_back_trace();                             \
        ASSERT(false);                                  \
    }                                                   \
} while(0)


#define NCCLCHECK(cmd) do {                             \
    ncclResult_t res = cmd;                             \
    if (res != ncclSuccess) {                           \
        printf("Failed, NCCL error %s:%d '%s'\n",       \
            __FILE__,__LINE__,ncclGetErrorString(res)); \
        print_back_trace();                             \
        ASSERT(false);                                  \
    }                                                   \
} while(0)


inline uintptr_t alloc_cuda_tensor(int count, int device_id, size_t size_of_item = 2) {
    ASSERT (count > 0);
    void* data;
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(device_id));
    #endif
    CUDACHECK(cudaMalloc(&data, count * size_of_item));
    return (uintptr_t) (data);
}

inline uintptr_t alloc_copy_tensor(uintptr_t buf, int size) {
    void* data;
    CUDACHECK(cudaMalloc(&data, size));
    CUDACHECK(cudaMemcpy(data, (void*) buf, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    return (uintptr_t) data;
}

inline void free_cuda_tensor(void *ptr) {
    CUDACHECK(cudaFree(ptr));
}

inline void* convert_to_cuda_buffer(size_t number) {
    void* data;
    CUDACHECK(cudaMalloc(&data, sizeof(size_t)));
    CUDACHECK(cudaMemcpy(data, &number, sizeof(size_t), cudaMemcpyHostToDevice));
    return data;
}

#ifdef D_ENABLE_NVTX

#include "nvtx3/nvtx3.hpp"

using tx_range = nvtx3::scoped_range;

#define AUTO_TX_RANGE tx_range __{__FUNCTION__}

#else

using tx_range = std::string;

#define AUTO_TX_RANGE

#endif

// TODO(hogura|20241001): add allocAsync