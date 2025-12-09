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

NcclChannel::NcclChannel(int party_local, int party_other, ncclUniqueId unique_id, cudaStream_t stream): 
    Channel::Channel(party_local, party_other), unique_id(unique_id) {
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

void NcclChannel::initialize() {
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->local));
    #endif
    NCCLCHECK(ncclCommInitRank(
        &this->comm,
        /*nranks=*/ 2,
        this->unique_id,
        /*rank=*/ this->m_rank()
    ));
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

namespace {
inline int queue_index(const BatchMetadata& metadata) {
    switch (metadata.batch_tag) {
        case BatchTag::ATTENTION: return 0;
        case BatchTag::EXPERT: return 1;
        case BatchTag::TOKENIZER: return 2;
        default: return 0;
    }
}
} // namespace

void TensorLocalChannel::send(uintptr_t data, const BatchMetadata& metadata) {
    int idx = queue_index(metadata);
    std::lock_guard<std::mutex> lock(m);
    data_buffers[idx].push(data);
    auto qsize = data_buffers[idx].size();
    c[idx].notify_one();
    if (qsize > 1024) {
        DMOE_LOG(WARNING) << "[TensorLocalChannel] queue backlog=" << qsize
                          << " elems for peer " << this->other
                          << " tag=" << static_cast<int>(metadata.batch_tag)
                          << LEND;
    }
}

void TensorLocalChannel::recv(uintptr_t data, const BatchMetadata& metadata) {
    std::unique_lock<std::mutex> lock(m);
    int idx = queue_index(metadata);
    auto t0 = t_now();
    while (data_buffers[idx].empty()) {
        c[idx].wait(lock);
    }
    auto waited = static_cast<long long>(t_now()) - static_cast<long long>(t0);
    if (waited > 5000) { // microseconds
        DMOE_LOG(WARNING) << "[TensorLocalChannel] recv waited " << waited
                          << "us for peer " << this->other
                          << " tag=" << static_cast<int>(metadata.batch_tag)
                          << LEND;
    }
    uintptr_t data_to_recv = data_buffers[idx].front();
    data_buffers[idx].pop();
    CUDACHECK(cudaMemcpyAsync((void *)data, (void*) data_to_recv, metadata.num_element() * metadata.get_datatype_size(), cudaMemcpyKind::cudaMemcpyDeviceToDevice, this->stream));
}

void TensorLocalChannel::sync() {
    CUDACHECK(cudaStreamSynchronize(this->stream));
}

std::mutex global_mutex;

Channel_t create_nccl_channel(int party_local, int party_other, ncclUniqueId unique_id) {
    auto channel = std::make_shared<NcclChannel>(party_local, party_other, unique_id);
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
