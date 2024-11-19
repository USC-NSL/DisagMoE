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

void NcclChannel::send(uintptr_t data_ptr, const Metadata& metadata) {
    DMOE_LOG(INFO) << "NCCL sending: " << local << " " << other << LEND;
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
    DMOE_LOG(INFO) << "NCCL sent " << local << " " << other << LEND;
}

void NcclChannel::recv(uintptr_t data_ptr, const Metadata& metadata) {
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

ZmqChannel::ZmqChannel(int party_local, int party_other, bool is_sender):
    Channel(party_local, party_other), is_sender(is_sender) {
        sprintf(device_id_str, "%d", party_local);
    }

std::map<int, mq_t> ZmqChannel::global_mq = {};
std::mutex global_mutex;

void ZmqChannel::instantiate() {
    DMOE_LOG(INFO) << "initiating zmq channel: " << local << " " << other << " " << is_sender << LEND;
    this->ctx = zmq::context_t(1);
    this->mq = std::make_shared<zmq::socket_t>(
        this->ctx, 
        this->is_sender ? zmq::socket_type::push : zmq::socket_type::pull
    );
    if (is_sender) {
        this->mq->bind(get_zmq_addr(local, /*is_gpu=*/ false));
    } else {
        this->mq->connect(get_zmq_addr(other, /*is_gpu=*/ false));
    }
    DMOE_LOG(INFO) << "ZmqChannel instantiated " << this->local << LEND;
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
    tx_range _{"ZmqChannel::send"};

    DMOE_LOG(DEBUG) << "ZmqChannel Sending to " << get_peer_id() << LEND;

    void* buf = this->_tensor_copy(data, metadata, /*to_gpu=*/ false);
    size_t size = metadata.num_element() * metadata.get_datatype_size();
    this->mq->send(zmq::buffer(buf, size));
    
    // !FIXME(hogura|20241009): may have memory leakage
    // if (data != (uintptr_t) buf)
    //     std::free(buf);

    DMOE_LOG(DEBUG) << "ZMQ Sent." << LEND;
}

void ZmqChannel::recv(uintptr_t data, const Metadata &metadata) {
    tx_range _{"ZmqChannel::recv"};

    DMOE_LOG(DEBUG) << "ZMQ Recving from " << get_peer_id() << LEND;

    size_t size = metadata.num_element() * metadata.get_datatype_size();
    zmq::message_t msg(size);
    auto err = this->mq->recv(msg, zmq::recv_flags::none);
    this->_tensor_copy((uintptr_t) msg.data(), metadata, 
        /*to_gpu=*/ !is_embedding_node(local), data);

    DMOE_LOG(DEBUG) << "ZMQ Recved" << LEND;
}

NcclGroupChannel::NcclGroupChannel(int party_local, const std::vector<int> &party_all, ncclUniqueId comm_id, cudaStream_t stream):
    NcclChannel(party_local, -1, comm_id, stream), size(party_all.size()) {
        #ifndef D_ENABLE_RAY
        CUDACHECK(cudaSetDevice(this->local));
        #endif
        if (!is_embedding_node(party_local)) {
            if (stream == nullptr) {
                CUDACHECK(cudaStreamCreateWithPriority(&this->stream, cudaStreamNonBlocking, 1));
            } else {
                this->stream = stream;
            }
        }
        local_rank = std::find(party_all.begin(), party_all.end(), party_local) - party_all.begin();
        ctx = zmq::context_t(1);
        mq = zmq::socket_t(ctx, is_root() ? zmq::socket_type::push : zmq::socket_type::pull);
        root_device_id = party_all[0];
        other = party_all[0];
        zmq_comm_id = 0;
        for (int i: party_all)
            zmq_comm_id += i;
        for (int i = 0; i < 128; i ++)
            (zmq_comm_id += comm_id.internal[i]) %= ZMQ_MAGIC_MOD;
    }

void NcclGroupChannel::instantiate() {
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->local));
    #endif
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    ncclCommInitRankConfig(
        &this->comm,
        /*nranks=*/ size,
        this->comm_id,
        /*rank=*/ this->local_rank,
        &config
    );
    ncclResult_t state;
    do {
        ncclCommGetAsyncError(comm, &state);
    } while(state == ncclInProgress);

    if (is_root())
        mq.bind(get_zmq_addr(root_device_id, false, /*manual_port=*/ ZMQ_GROUP_PORT + zmq_comm_id));
    else
        mq.connect(get_zmq_addr(root_device_id, false, /*manual_port=*/ ZMQ_GROUP_PORT + zmq_comm_id));
}

bool NcclGroupChannel::is_root() const {
    return this->local_rank == root();
}

int NcclGroupChannel::root() const {
    return 0;
}

void NcclGroupChannel::broadcast(void* send_buf, void* recv_buf, size_t count, ncclDataType_t type, cudaStream_t stream) {
    tx_range _{"NcclGroupChannel::broadcast"};
    DMOE_LOG(DEBUG) << "broadcasting " << root() << " " << local << " " << count << " on the stream " << this->stream << LEND;
    NCCLCHECK(ncclBroadcast(
        send_buf,
        recv_buf,
        count,
        type,
        root(),
        this->comm,
        this->stream
    ));
    CUDACHECK(cudaStreamSynchronize(this->stream));
    DMOE_LOG(DEBUG) << "finished broadcast" << root() << " " << local << " " << count << " on the stream " << this->stream << LEND;
}

void NcclGroupChannel::send(uintptr_t data_ptr, const Metadata& metadata) {
    tx_range _{"NcclGroupChannel::send"};
    ASSERT(is_root());
    DMOE_LOG(DEBUG) << "NcclGroupChannel Sending from " << this->local << LEND;
    send_recv(data_ptr, metadata);
    DMOE_LOG(DEBUG) << "NcclGroupChannel Sent." << LEND;
}

void NcclGroupChannel::recv(uintptr_t data_ptr, const Metadata& metadata) {
    tx_range _{"NcclGroupChannel::recv"};
    ASSERT(!is_root());
    DMOE_LOG(DEBUG) << "NcclGroupChannel Recving from " << this->local << LEND;
    send_recv(data_ptr, metadata);
    DMOE_LOG(DEBUG) << "NcclGroupChannel Recved." << LEND;
}

void NcclGroupChannel::send_recv(uintptr_t data_ptr, const Metadata& metadata) {
    broadcast(reinterpret_cast<void*>(data_ptr), reinterpret_cast<void*>(data_ptr), 
        metadata.num_element(), metadata.get_nccl_datatype());
}

void NcclGroupChannel::bcast_obj(void* &buf, size_t &size, cudaStream_t stream) {
    tx_range _{"NcclGroupChannel::bcast_obj"};
    if (is_root()) {
        void* data_buf = (void*) alloc_cuda_tensor(size, this->local, /*size_of_item=*/ sizeof(char));
        CUDACHECK(cudaMemcpy(data_buf, buf, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        // first send size
        // use zmq to broadcast
        for (int i = 0; i < this->size - 1; i ++)
            this->mq.send(zmq::buffer((char*) &size, sizeof(size_t)));
        // then send data
        broadcast(data_buf, data_buf, size, ncclInt8);
        free_cuda_tensor(data_buf);
    } else {
        // first recv size
        zmq::message_t msg(sizeof(size));
        this->mq.recv(msg, zmq::recv_flags::none);
        size = *(size_t*) msg.data();
        DMOE_LOG(DEBUG) << "recved size: " << size << LEND;
        ASSERT(size > 0);
        // then recv data
        void* data_buf = (void*) alloc_cuda_tensor(size, this->local, /*size_of_item=*/ sizeof(char));
        broadcast(data_buf, data_buf, size, ncclInt8);
        buf = std::malloc(size);
        CUDACHECK(cudaMemcpy(buf, data_buf, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        // DMOE_LOG(DEBUG) << "received metadata " << *decerealize<Metadata>((char*) buf, size) << LEND;
        free_cuda_tensor(data_buf);
    }
}

void NcclGroupChannel::send_metadata(const Metadata& metadata) {
    tx_range _{"NcclGroupChannel::bcast_metadata"};
    ASSERT(is_root());
    DMOE_LOG(DEBUG) << "NcclGroupChannel Sending metadata from " << this->local << ":" << metadata << LEND;
    std::string data = cerealize(std::make_shared<Metadata>(metadata));
    void* buf = (void*) data.data();
    size_t size = data.size();
    bcast_obj(buf, size);
    DMOE_LOG(DEBUG) << "NcclGroupChannel Sent metadata." << LEND;
}

void NcclGroupChannel::recv_metadata(Metadata& metadata) {
    tx_range _{"NcclGroupChannel::recv_metadata"};
    ASSERT(!is_root());
    DMOE_LOG(DEBUG) << "NcclGroupChannel Recving metadata from " << this->local << LEND;
    void* buf;
    size_t size;
    bcast_obj(buf, size);

    metadata = *decerealize<Metadata>((char*) buf, size);
    std::free(buf);
    DMOE_LOG(DEBUG) << "NcclGroupChannel Recved metadata." << LEND;
}

void NcclGroupChannel::all_reduce(uintptr_t data, const std::vector<int> &shape) {
    tx_range _{"NcclGroupChannel::all_reduce"};
    void* buf = reinterpret_cast<void*>(data);
    int count = 1;
    for (int i: shape)
        count *= i;
    DMOE_LOG(DEBUG) << "Calling all_reduce for " << count << " elements." << LEND;
    NCCLCHECK(ncclAllReduce(
        buf,
        buf,
        count,
        ncclBfloat16,   // !FIXME(hogura|20241106): remove this hardcode
        ncclSum,
        this->comm,
        this->stream
    ));
    CUDACHECK(cudaStreamSynchronize(this->stream));
    DMOE_LOG(DEBUG) << "AllReduce done." << LEND;
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

Channel_t create_nccl_group_channel(int party_local, const std::vector<int> &party_all, void *nccl_id_raw) {
    ncclUniqueId& id = *((ncclUniqueId*)(nccl_id_raw));
    auto channel = std::make_shared<NcclGroupChannel>(
        party_local, party_all, id
    );
    // TODO(hogura|20241103): recycle the ncclUniqueId (raw).
    return channel;
}

std::vector<Channel_t> create_nccl_group_channels(int root, const std::vector<int> &party_all, void *nccl_id_raw) {
    ASSERT(root == 0);
    ncclUniqueId& id = *((ncclUniqueId*)(nccl_id_raw));
    std::vector<Channel_t> channels;
    for (int party_local: party_all) {
        channels.push_back(std::make_shared<NcclGroupChannel>(
            party_local, party_all, id
        ));
    }
    // TODO(hogura|20241103): recycle the ncclUniqueId (raw).
    return channels;
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
