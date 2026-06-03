#include "comm.h"
#include "logging.h"
#include "utils.hpp"
#include "distributed.hpp"
#include "metadata.hpp"
#include "batch.hpp"

#if USE_NIXL
#include "nixl_channel.h"
#endif

#include <iomanip>
#include <mutex>
#include <memory>
#include <cstring>
#include <cstdlib>

// static std::shared_ptr<std::mutex> global_mutex = std::make_shared<std::mutex>();
// NOTE: global_mutex was removed — it caused deadlocks between MuDispatcher (ncclSend)
// and MuPool (ncclRecv) when NCCL's lazy P2P transport setup required both sides to
// call their respective ops simultaneously. Each NcclChannel has its own ncclComm_t
// and cudaStream_t, so concurrent ops on different channels are safe per NCCL docs.

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
 
void NcclChannel::send_raw(uintptr_t data_ptr, const BatchMetadata& metadata) {
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

void NcclChannel::recv_raw(uintptr_t data_ptr, const BatchMetadata& metadata) {
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

void NcclChannel::send_batch(const torch::Tensor &data, const BatchMetadata& metadata) {
    this->send_raw((uintptr_t)data.data_ptr(), metadata);
}

void NcclChannel::recv_batch(const torch::Tensor &data, const BatchMetadata& metadata) {
    this->recv_raw((uintptr_t)data.data_ptr(), metadata);
}

void NcclChannel::warmup_send(int *send_buf, int count) {
    // do a simple all reduce to warm up the NCCL communication
    NCCLCHECK(ncclSend(
        send_buf,
        /*count=*/ count,
        /*datatype=*/ ncclInt,
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
}

void NcclChannel::warmup_recv(int *recv_buf, int count) {
    // do a simple all reduce to warm up the NCCL communication
    NCCLCHECK(ncclRecv(
        recv_buf,
        /*count=*/ count,
        /*datatype=*/ ncclInt,
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
}

void NcclChannel::record_event(cudaEvent_t &event) {
    CUDACHECK(cudaEventRecord(event, this->stream));
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

void TensorLocalChannel::send_raw(uintptr_t data, const BatchMetadata& metadata) {
    std::lock_guard<std::mutex> lock(m);
    data_buffer.push(data);
    c.notify_one();
}

void TensorLocalChannel::recv_raw(uintptr_t data, const BatchMetadata& metadata) {
    std::unique_lock<std::mutex> lock(m);
    while (data_buffer.empty()) {
        c.wait(lock);
    }
    uintptr_t data_to_recv = data_buffer.front();
    data_buffer.pop();
    cudaMemcpyAsync((void *)data, (void*) data_to_recv, metadata.num_element() * metadata.get_datatype_size(), cudaMemcpyKind::cudaMemcpyDeviceToDevice, this->stream);
}

void TensorLocalChannel::send_batch(const torch::Tensor &data, const BatchMetadata& metadata) {
    std::lock_guard<std::mutex> lock(m);
    batch_buffer.push(data);
    c.notify_one();
}

void TensorLocalChannel::recv_batch(const torch::Tensor &data, const BatchMetadata& metadata) {
    std::unique_lock<std::mutex> lock(m);
    while (batch_buffer.empty()) {
        c.wait(lock);
    }
    torch::Tensor data_to_recv = batch_buffer.front();
    cudaMemcpyAsync((void *)data.data_ptr(), (void*) data_to_recv.data_ptr(), metadata.num_element() * metadata.get_datatype_size(), cudaMemcpyKind::cudaMemcpyDeviceToDevice, this->stream);
    batch_buffer.pop();
}

void TensorLocalChannel::sync() {
    CUDACHECK(cudaStreamSynchronize(this->stream));
}

void TensorLocalChannel::record_event(cudaEvent_t &event) {
    CUDACHECK(cudaEventRecord(event, this->stream));
}

Channel_t create_nccl_channel(int party_local, int party_other, ncclUniqueId unique_id) {
    auto channel = std::make_shared<NcclChannel>(party_local, party_other, unique_id);
    return channel;
}

Channel_t create_nixl_channel(int party_local, int party_other) {
#if USE_NIXL
    return std::make_shared<NixlChannel>(party_local, party_other);
#else
    (void)party_local;
    (void)party_other;
    throw std::runtime_error("create_nixl_channel called when USE_NIXL=0");
#endif
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
