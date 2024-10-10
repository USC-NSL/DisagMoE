#include "comm.h"
#include "logging.h"
#include "utils.hpp"

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
    // debug_print_environ();
    printf("calling nccl init (%d, %d) %d\n", this->local, this->other, this->m_rank());
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->local));
    #endif
    NCCLCHECK(ncclCommInitRank(
        &this->comm,
        /*nranks=*/ 2,
        this->comm_id,
        /*rank=*/ this->m_rank()
    ));
    puts("called nccl init");
}

void NcclChannel::send(uintptr_t data_ptr, const Metadata& metadata) {
    // TODO(hogura|20240926): check if ncclGroupStart may influence the performance
    void* data = reinterpret_cast<void*>(data_ptr);
    printf("NCCL sending %u device=%d\n", metadata.num_element(), this->local);
    NCCLCHECK(ncclSend(
        data, 
        /*count=*/ metadata.num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
    printf("%d NCCL sent\n", this->local);
}

void NcclChannel::recv(uintptr_t data_ptr, const Metadata& metadata) {
    void* data = reinterpret_cast<void*>(data_ptr);
    printf("NCCL recving %u device=%d\n", metadata.num_element(), this->local);
    NCCLCHECK(ncclRecv(
        data,
        /*count=*/ metadata.num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
    printf("%d NCCL recved\n", this->local);
}

ZmqChannel::ZmqChannel(int party_local, int party_other, bool is_sender):
    Channel(party_local, party_other), is_sender(is_sender) {
        sprintf(device_id_str, "%d", party_local);
    }

std::map<int, mq_t> ZmqChannel::global_mq = {};
std::mutex global_mutex;

void ZmqChannel::instantiate() {
    // !FIXME(hogura|20241009): this type of sharing mq may have issues.
    LOG(INFO) << "initiating zmq channel: " << local << " " << other << " " << is_sender << LEND;
    // std::lock_guard<std::mutex> guard(global_mutex);
    // if (global_mq.find(this->local) != global_mq.end()) {
    //     this->mq = global_mq[this->local];
    //     LOG(INFO) << "ZmqChannel instantiated " << this->local << LEND;
    //     return;
    // }
    this->ctx = zmq::context_t(1);
    this->mq = std::make_shared<zmq::socket_t>(
        this->ctx, 
        this->is_sender ? zmq::socket_type::push : zmq::socket_type::pull
    );
    // global_mq[this->local] = this->mq;
    if (is_sender) {
        this->mq->bind(get_zmq_addr(local, /*is_gpu=*/ false));
    } else {
        this->mq->connect(get_zmq_addr(other, /*is_gpu=*/ false));
    }
    LOG(INFO) << "ZmqChannel instantiated " << this->local << LEND;
}

void* ZmqChannel::_tensor_copy(uintptr_t data, const Metadata& metadata, bool to_gpu, uintptr_t dst) {
    if (is_embedding_node(this->local))
        return (void*) data;
    size_t size = metadata.num_element() * metadata.get_datatype_size();
    uintptr_t buf;
    if (!to_gpu) {
        buf = !dst ? (uintptr_t) std::malloc(size) : dst;
        // TODO(hogura|20241007): overlap this memcpy
        CUDACHECK(cudaMemcpy((void*) buf, (void*) data, size, 
            cudaMemcpyKind::cudaMemcpyDeviceToHost));
    } else {
        buf = !dst ? alloc_cuda_tensor(metadata.num_element(), this->local) : dst;
        CUDACHECK(cudaMemcpy((void*) buf, (void*) data, size, 
            cudaMemcpyKind::cudaMemcpyHostToDevice));
    }
    return (void*) buf;
}

void ZmqChannel::send(uintptr_t data, const Metadata& metadata) {
    LOG(INFO) << local << "ZmqChannel sending " << metadata << LEND;
    void* buf = this->_tensor_copy(data, metadata, /*to_gpu=*/ false);
    size_t size = metadata.num_element() * metadata.get_datatype_size();
    print_buf(buf, size);
    this->mq->send(zmq::buffer(buf, size));
    LOG(INFO) << local << "ZmqChannel sent" << LEND;
    // !FIXME(hogura|20241009): may have memory leakage
    if (data != (uintptr_t) buf)
        std::free(buf);
}

void ZmqChannel::recv(uintptr_t data, const Metadata &metadata) {
    LOG(INFO) << local << "ZmqChannel recving " << metadata << LEND;
    size_t size = metadata.num_element() * metadata.get_datatype_size();
    zmq::message_t msg(size);
    auto err = this->mq->recv(msg, zmq::recv_flags::none);
    LOG(INFO) << local << "ZmqChannel recved, now copying data" << LEND;
    print_buf(msg.data(), size);
    this->_tensor_copy((uintptr_t) msg.data(), metadata, /*to_gpu=*/ !is_embedding_node(local), data);
    LOG(INFO) << local << "ZmqChannel recv ended" << LEND;
}

Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw) {
    ncclUniqueId& id = *((ncclUniqueId*)(nccl_id_raw));
    auto channel = std::make_shared<NcclChannel>(
        party_local, party_other, id
    );
    // TODO(hogura|20240927): recycle the ncclUniqueId (raw).
    return channel;
}

Channel_t create_zmq_channel(int party_local, int party_other, bool is_sender) {
    auto channel = std::make_shared<ZmqChannel>(party_local, party_other, is_sender);
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