#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <cstdio>

ncclUniqueId id;

void func(int rank) {
    cudaSetDevice(rank);
    ncclComm_t comm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    ncclCommInitRankConfig(&comm, 2, id, rank, &config);
    ncclResult_t state;
    do {
        ncclCommGetAsyncError(comm, &state);
    } while(state == ncclInProgress);

    printf("init %d\n", rank);

    float* buf;
    cudaMalloc(&buf, sizeof(float));
    cudaMemset(buf, 0, sizeof(float));
    ncclAllReduce(buf, buf, 1, ncclFloat, ncclSum, comm, cudaStreamDefault);
    do {
        ncclCommGetAsyncError(comm, &state);
    } while(state == ncclInProgress);

    printf("allreduce %d\n", rank);
}

int main() {
    ncclGetUniqueId(&id);
    std::thread t1(func, 0);
    std::thread t2(func, 1);
    t1.join();
    t2.join();
}