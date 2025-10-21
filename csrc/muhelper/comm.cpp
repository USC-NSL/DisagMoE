#include "comm.h"
#include "logging.h"
#include "utils.hpp"
#include "distributed.hpp"
#include "metadata.hpp"
#include "batch.hpp"

#include <iomanip>
#include <mutex>

NcclChannel::NcclChannel(int party_local, int party_other, ncclUniqueId comm_id, cudaStream_t stream): 
    Channel::Channel(party_local, party_other), comm_id(comm_id) 
    {
        // TODO(hogura|20240927): convert the party_local to local gpu rank (0<local<num_gpu)
        #ifndef D_ENABLE_RAY
        CUDACHECK(cudaSetDevice(this->local));
        #endif
        if (!is_embedding_node(party_local)) {
            if (stream == nullptr) {
                CUDACHECK(cudaStreamCreate(&this->stream));
                // CUDACHECK(cudaStreamCreateWithPriority(&this->stream, cudaStreamNonBlocking, 1));
            } else {
                this->stream = stream;
            }
        }
    }

NcclChannel::~NcclChannel() {
    // NCCLCHECK(ncclCommFinalize(this->comm));
    // NCCLCHECK(ncclCommDestroy(this->comm));
}

extern char** _environ;
void debug_print_environ() {
    puts("Printing environ");
    for (char** s = _environ; *s; s++) {
        printf("%s\n", *s);
    }
}

void NcclChannel::instantiate() {
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->local));
    #endif
    NCCLCHECK(ncclCommInitRank(
        &this->comm,
        /*nranks=*/ 2,
        this->comm_id,
        /*rank=*/ this->m_rank()
    ));
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

TensorLocalChannel::~TensorLocalChannel() {
    // CUDACHECK(cudaStreamDestroy(this->stream));
}

void TensorLocalChannel::instantiate() {
    // do nothing
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

ZmqChannel::ZmqChannel(int party_local, int party_other, bool is_sender, int rank):
    Channel(party_local, party_other), is_sender(is_sender), rank_offset(rank) {
        sprintf(device_id_str, "%d", party_local);
        if (!is_embedding_node(party_local)) {
            CUDACHECK(cudaStreamCreateWithPriority(&this->stream, cudaStreamNonBlocking, 10));
        } else {
            this->stream = 0;
        }
    }

std::map<int, mq_t> ZmqChannel::global_mq = {};
std::mutex global_mutex;

void ZmqChannel::instantiate() {
    // DMOE_LOG(INFO) << "initiating zmq channel: " << local << " " << other << " " << is_sender << " " << this->rank_offset << LEND;
    this->ctx = zmq::context_t(1);
    this->mq = std::make_shared<zmq::socket_t>(
        this->ctx, 
        this->is_sender ? zmq::socket_type::push : zmq::socket_type::pull
    );
    std::string addr;
    if (is_sender) {
        addr = get_zmq_addr(local, /*is_gpu=*/ false, /*manual_port=*/ -1, /*offset=*/ this->rank_offset);
        this->mq->bind(addr);
    } else {
        addr = get_zmq_addr(other, /*is_gpu=*/ false, /*manual_port=*/ -1, /*offset=*/ this->rank_offset);
        this->mq->connect(addr);
    }
    // DMOE_LOG(WARNING) << "ZmqChannel instantiated, local: " << this->local << ", remote: " << this->other << ", addr: " << addr << LEND;
}

void* ZmqChannel::_tensor_copy(uintptr_t data, const BatchMetadata& metadata, bool to_gpu, uintptr_t dst) {
    if (is_embedding_node(this->local))
        return (void*) data;
    tx_range _{"ZmqChannel::_tensor_copy"};
    uintptr_t buf;
    cudaMemcpyKind flag;
    if (!to_gpu) {
        size_t size = metadata.num_element() * metadata.get_datatype_size();
        buf = !dst ? (uintptr_t) std::malloc(size) : dst;
        flag = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    } else {
        buf = dst;
        flag = cudaMemcpyKind::cudaMemcpyHostToDevice;
    }

    {
        tx_range __{"ZmqChannel::_tensor_copy_memcpy_submit"};
        const size_t step = 4;
        const size_t dim_stride = metadata.get_datatype_size() * metadata.token_hidden_dim();
        size_t num_tokens = metadata.shape[0];
        for (size_t i = 0; i < metadata.shape[0]; i += step) {
            size_t cur_step = std::min(step, metadata.shape[0] - i);
            CUDACHECK(cudaMemcpyAsync(
                (void*) (buf + i * dim_stride),
                (void*) (data + i * dim_stride),
                cur_step * dim_stride,
                flag,
                this->stream
            ));
        }
    }
    CUDACHECK(cudaStreamSynchronize(this->stream));

    return (void*) buf;
}

void ZmqChannel::send(uintptr_t data, const BatchMetadata& metadata) {
    tx_range _{"ZmqChannel::send"};

    // DMOE_LOG(DEBUG) << "ZmqChannel Sending to " << get_peer_id() << LEND;
    std::vector<int> token_ids(metadata.num_tokens(), 0);
    size_t size = metadata.num_tokens() * sizeof(int);
    // DMOE_LOG(DEBUG) << "send size: " << size << " rank: " << this->rank_offset << LEND;
    this->mq->send(zmq::buffer(token_ids.data(), size));
    
    // if (data != (uintptr_t) buf)
    //     std::free(buf);

    // DMOE_LOG(DEBUG) << "ZMQ Sent." << LEND;
}

void ZmqChannel::recv(uintptr_t data, const BatchMetadata &metadata) {
    tx_range _{"ZmqChannel::recv"};

    // DMOE_LOG(DEBUG) << "ZMQ Recving from " << get_peer_id() << LEND;

    size_t size = metadata.num_tokens() * sizeof(int);
    zmq::message_t msg(size);
    // DMOE_LOG(DEBUG) << "recv size: " << size << " rank: " << this->rank_offset << LEND;
    auto err = this->mq->recv(msg, zmq::recv_flags::none);
    // DMOE_LOG(DEBUG) << "ZMQ Recved" << LEND;
}

Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw) {
    ncclUniqueId& id = *((ncclUniqueId*)(nccl_id_raw));
    auto channel = std::make_shared<NcclChannel>(
        party_local, party_other, id
    );
    // TODO(hogura|20240927): recycle the ncclUniqueId (raw).
    return channel;
}

Channel_t create_local_channel(int device_id) {
    auto channel = std::make_shared<TensorLocalChannel>(device_id);
    return channel;
}

Channel_t create_zmq_channel(int party_local, int party_other, bool is_sender, int rank) {
    auto channel = std::make_shared<ZmqChannel>(party_local, party_other, is_sender, rank);
    return channel;
}

void* get_nccl_unique_id() {
    void* _data = std::malloc(sizeof(ncclUniqueId));
    ncclGetUniqueId((ncclUniqueId*)_data);
    return _data;
}

void instantiate_channels(std::vector<Channel_t> channels) {
    std::vector<std::thread> threads;
    puts("creating channels");
    for (auto c: channels) {
        c->_debug_print();
        threads.push_back(std::thread([=](auto channel) {channel->instantiate();}, c));
    }
    for (auto &t: threads) {
        t.join();
    }
    puts("threads inited");
}
