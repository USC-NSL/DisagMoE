#include "comm.h"
#include "logging.h"
#include "utils.hpp"
#include "distributed.hpp"
#include "metadata.hpp"
#include "batch.hpp"

#include <iomanip>
#include <mutex>
#include <cstring>
#include <cstdlib>

NcclChannel::NcclChannel(int party_local, int party_other, ncclComm_t comm, cudaStream_t stream): 
    Channel::Channel(party_local, party_other), comm(comm) {
    // TODO(hogura|20240927): convert the party_local to local gpu rank (0<local<num_gpu)
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->local));
    #endif
    if (stream == nullptr) {
        CUDACHECK(cudaStreamCreate(&this->stream));
        // CUDACHECK(cudaStreamCreateWithPriority(&this->stream, cudaStreamNonBlocking, 1));
    } else {
        this->stream = stream;
    }
}

extern char** _environ;
void debug_print_environ() {
    puts("Printing environ");
    for (char** s = _environ; *s; s++) {
        printf("%s\n", *s);
    }
}

void NcclChannel::send(uintptr_t data_ptr, const BatchMetadata& metadata) {
    // DMOE_LOG(INFO) << "NCCL sending: " << local << " " << other << LEND;
    tx_range _{"NcclChannel::send"};
    void* data = reinterpret_cast<void*>(data_ptr);
    NCCLCHECK(ncclSend(
        data, 
        /*count=*/ metadata.num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
    // CUDACHECK(cudaStreamSynchronize(this->stream));
    // DMOE_LOG(INFO) << "NCCL sent " << local << " " << other << LEND;
}

void NcclChannel::recv(uintptr_t data_ptr, const BatchMetadata& metadata) {
    tx_range _{"NcclChannel::recv"};
    void* data = reinterpret_cast<void*>(data_ptr);
    NCCLCHECK(ncclRecv(
        data,
        /*count=*/ metadata.num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
}

void NcclChannel::sync() {
    CUDACHECK(cudaStreamSynchronize(this->stream));
}

TensorLocalChannel::TensorLocalChannel(int device_id, cudaStream_t stream):
    Channel(device_id, device_id), stream(stream) {
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->local));
    #endif
    if (stream == nullptr) {
        CUDACHECK(cudaStreamCreate(&this->stream));
    } 
}

void TensorLocalChannel::send(uintptr_t data, const BatchMetadata& metadata) {
    std::lock_guard<std::mutex> lock(m);
    data_buffer.push(data);
    c.notify_one();
}

void TensorLocalChannel::recv(uintptr_t data, const BatchMetadata& metadata) {
    std::unique_lock<std::mutex> lock(m);
    while (data_buffer.empty()) {
        c.wait(lock);
    }
    uintptr_t data_to_recv = data_buffer.front();
    data_buffer.pop();
    cudaMemcpy((void *)data, (void*) data_to_recv, metadata.num_element() * metadata.get_datatype_size(), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void TensorLocalChannel::sync() {
    CUDACHECK(cudaStreamSynchronize(this->stream));
}

std::mutex global_mutex;

Channel_t create_nccl_channel(int party_local, int party_other, ncclComm_t comm) {
    auto channel = std::make_shared<NcclChannel>(party_local, party_other, comm);
    return channel;
}

Channel_t create_local_channel(int device_id) {
    auto channel = std::make_shared<TensorLocalChannel>(device_id);
    return channel;
}

void* get_nccl_unique_id() {
    void* _data = std::malloc(sizeof(ncclUniqueId));
    ncclGetUniqueId((ncclUniqueId*)_data);
    return _data;
}