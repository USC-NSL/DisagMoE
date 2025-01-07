#include "comm.h"
#include "logging.h"
#include "utils.hpp"
#include "distributed.hpp"

#include <iomanip>
#include <mutex>

using Tensor = torch::Tensor;

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

void NcclChannel::_delay_release_tensor(Tensor tensor, cudaStream_t stream) {
    struct TensorWarp {
        Tensor tensor;
    };
    TensorWarp* warp = new TensorWarp{std::move(tensor)};
    CUDACHECK(cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* data) {
        TensorWarp* warp = (TensorWarp*) data;
        delete warp;
    }, (void*) warp, 0));
}

void NcclChannel::send(Tensor tensor, const Metadata& metadata) {
    // DMOE_LOG(INFO) << "NCCL sending: " << local << " " << other << LEND;
    tx_range _{"NcclChannel::send"};
    void* data = reinterpret_cast<void*>(tensor.data_ptr());
    NCCLCHECK(ncclSend(
        data,
        /*count=*/ metadata.num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
    _delay_release_tensor(tensor, this->stream);
    // DMOE_LOG(INFO) << "NCCL sent " << local << " " << other << LEND;
}

void NcclChannel::recv(Tensor tensor, const Metadata& metadata) {
    tx_range _{"NcclChannel::recv"};
    void* data = reinterpret_cast<void*>(tensor.data_ptr());
    NCCLCHECK(ncclRecv(
        data,
        /*count=*/ metadata.num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        /*peer=*/ this->m_other(),
        this->comm,
        this->stream
    ));
    CUDACHECK(cudaStreamSynchronize(this->stream));
}

void NcclChannel::sync() {
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
    DMOE_LOG(INFO) << "initiating zmq channel: " << local << " " << other << " " << is_sender << " " << this->rank_offset << LEND;
    this->ctx = zmq::context_t(1);
    this->mq = std::make_shared<zmq::socket_t>(
        this->ctx, 
        this->is_sender ? zmq::socket_type::push : zmq::socket_type::pull
    );
    if (is_sender) {
        this->mq->bind(get_zmq_addr(local, /*is_gpu=*/ false, /*manual_port=*/ -1, /*offset=*/ this->rank_offset));
    } else {
        this->mq->connect(get_zmq_addr(other, /*is_gpu=*/ false, /*manual_port=*/ -1, /*offset=*/ this->rank_offset));
    }
    DMOE_LOG(INFO) << "ZmqChannel instantiated " << this->local << LEND;

    // CUDACHECK(cudaMallocHost(&this->pin_buffer, 512 * 4096 * 4));
    this->pin_buffer = (void*) std::malloc(512 * 4096 * 4);
}

void* ZmqChannel::_tensor_copy(uintptr_t data, const Metadata& metadata, bool to_gpu, uintptr_t dst) {
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
        const size_t step = 2;
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

void ZmqChannel::_pipeline_comm(void* data, size_t num_tokens, size_t token_size, cudaMemcpyKind flag) {
    tx_range _{"ZmqChannel::_pipeline_comm"};

    const size_t step = PIPE_COMM_STEP;
    void* cpu_buf = std::malloc(num_tokens * token_size);
    for (size_t i = 0; i < num_tokens; i += step) {
        size_t cur_step = std::min(step, num_tokens - i);

        void* src_buf = data + i * token_size;
        void* dst_buf = cpu_buf;
        if (flag == cudaMemcpyHostToDevice) {
            std::swap(src_buf, dst_buf);
            zmq::message_t msg(src_buf, cur_step * token_size);
            this->mq->recv(msg);
        }

        CUDACHECK(cudaMemcpyAsync(
            dst_buf,
            src_buf,
            cur_step * token_size,
            flag,
            this->stream
        ));

        if (flag == cudaMemcpyDeviceToHost) {
            // TODO(hogura|20250106): optimize this stream
            CUDACHECK(cudaStreamSynchronize(this->stream));
            zmq::message_t msg(dst_buf, cur_step * token_size);
            this->mq->send(msg);
        }
    }
    std::free(cpu_buf);
    cudaStreamSynchronize(this->stream);
}

void ZmqChannel::send(Tensor tensor, const Metadata& metadata) {
    tx_range _{"ZmqChannel::send"};

    // DMOE_LOG(DEBUG) << "ZmqChannel Sending to " << get_peer_id() << LEND;

    if (is_embedding_node(this->local)) {
        const int step = PIPE_COMM_STEP;
        void* buf = (void*) tensor.data_ptr();
        size_t size = metadata.num_element() * metadata.get_datatype_size();
        size_t step_size = step * metadata.token_hidden_dim() * metadata.get_datatype_size();
        for (size_t i = 0; i < size; i += step_size) {
            this->mq->send(zmq::buffer(buf + i, std::min(step_size, size - i)));
        }
    } else {
        this->_pipeline_comm(
            (void*) tensor.data_ptr(),
            metadata.num_tokens(),
            metadata.token_hidden_dim() * metadata.get_datatype_size(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost
        );
    }

    // DMOE_LOG(DEBUG) << "ZMQ Sent." << LEND;
}

void ZmqChannel::recv(Tensor tensor, const Metadata &metadata) {
    tx_range _{"ZmqChannel::recv"};

    // DMOE_LOG(DEBUG) << "ZMQ Recving from " << get_peer_id() << LEND;

    if (is_embedding_node(this->local)) {
        size_t size = metadata.num_element() * metadata.get_datatype_size();
        size_t step_size = PIPE_COMM_STEP * metadata.token_hidden_dim() * metadata.get_datatype_size();
        zmq::message_t msg(size);
        void* buf = (void*) tensor.data_ptr();
        for (size_t i = 0; i < size; i += step_size) {
            auto err = this->mq->recv(msg, zmq::recv_flags::none);
            memcpy(buf + i, msg.data(), std::min(step_size, size - i));
        }
    } else {
        this->_pipeline_comm(
            (void*) tensor.data_ptr(),
            metadata.num_tokens(),
            metadata.token_hidden_dim() * metadata.get_datatype_size(),
            cudaMemcpyKind::cudaMemcpyHostToDevice
        );
    }

    // DMOE_LOG(DEBUG) << "ZMQ Recved" << LEND;
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
        ASSERT(0 <= local_rank && local_rank < size);
        ctx = zmq::context_t(1);
        mq = zmq::socket_t(ctx, is_root() ? zmq::socket_type::push : zmq::socket_type::pull);
        root_device_id = party_all[0];
        other = party_all[0];
        zmq_comm_id = 0;
        for (int i: party_all)
            zmq_comm_id += i;
        for (int i = 0; i < 128; i ++)
            (zmq_comm_id += comm_id.internal[i]) %= ZMQ_MAGIC_MOD;
        
        // use barrier as a signal to synchronize in broadcast
        CUDACHECK(cudaMalloc(&barrier, sizeof(int)));
        CUDACHECK(cudaMemset(barrier, 0, sizeof(int)));
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
    
    this->buffer_gpu = torch::empty(
        {GROUP_CHANNEL_BUFFER_SIZE}, 
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA, 0)
    );

    synchronize();

    if (is_root())
        mq.bind(get_zmq_addr(root_device_id, false, /*manual_port=*/ ZMQ_GROUP_PORT + zmq_comm_id));
    else
        mq.connect(get_zmq_addr(root_device_id, false, /*manual_port=*/ ZMQ_GROUP_PORT + zmq_comm_id));
}

void NcclGroupChannel::synchronize() {
    NCCLCHECK(ncclAllReduce(barrier, barrier, 1, ncclInt, ncclSum, this->comm, this->stream));
    CUDACHECK(cudaStreamSynchronize(this->stream));
    DMOE_LOG(DEBUG) << "group_channel synchronized " << local << " " << local_rank << " " << this->stream << LEND;
}

bool NcclGroupChannel::is_root() const {
    return this->local_rank == root();
}

int NcclGroupChannel::root() const {
    return 0;
}

void NcclGroupChannel::broadcast(Tensor send_tensor, Tensor recv_tensor, size_t count, ncclDataType_t type, cudaStream_t stream) {
    tx_range _{"NcclGroupChannel::broadcast"};
    // DMOE_LOG(DEBUG) << "broadcasting " << root() << " " << local_rank << " " << count << " on the stream " << this->stream << LEND;
    NCCLCHECK(ncclBroadcast(
        (void*) send_tensor.data_ptr(),
        (void*) recv_tensor.data_ptr(),
        count,
        type,
        root(),
        this->comm,
        this->stream
    ));
    CUDACHECK(cudaStreamSynchronize(this->stream));
    // DMOE_LOG(DEBUG) << "finished broadcast " << root() << " " << local_rank << " " << count << " on the stream " << this->stream << LEND;
}

void NcclGroupChannel::send(Tensor tensor, const Metadata& metadata) {
    tx_range _{"NcclGroupChannel::send"};
    ASSERT(is_root());
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Sending from " << this->local << LEND;
    send_recv(std::move(tensor), metadata);
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Sent." << LEND;
}

void NcclGroupChannel::recv(Tensor tensor, const Metadata& metadata) {
    tx_range _{"NcclGroupChannel::recv"};
    ASSERT(!is_root());
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Recving from " << this->local << LEND;
    send_recv(std::move(tensor), metadata);
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Recved." << LEND;
}

void NcclGroupChannel::send_recv(Tensor tensor, const Metadata& metadata) {
    broadcast(tensor, tensor, metadata.num_element(), metadata.get_nccl_datatype());
}

void NcclGroupChannel::bcast_obj(void* &buf, size_t &size) {
    tx_range _{"NcclGroupChannel::bcast_obj"};
    if (is_root()) {
        // first send size
        // [option 1] use zmq to broadcast
        for (int i = 0; i < this->size - 1; i ++)
            this->mq.send(zmq::buffer((char*) &size, sizeof(size_t)));
        // DMOE_LOG(DEBUG) << "sent size: " << size << LEND;

        // [option 2] use NcclBroadcast
        // CUDACHECK(cudaMemcpyAsync(
        //     this->buffer_gpu.data_ptr(), &size, 
        //     sizeof(size_t), 
        //     cudaMemcpyKind::cudaMemcpyHostToDevice,
        //     this->stream
        // ));
        // broadcast(this->buffer_gpu.data_ptr(), this->buffer_gpu.data_ptr(), 1, ncclUint64);

        // then send data
        ASSERT (size <= GROUP_CHANNEL_BUFFER_SIZE);
        void* data_buf = (void*) this->buffer_gpu.data_ptr();
        CUDACHECK(cudaMemcpy(data_buf, buf, size, cudaMemcpyKind::cudaMemcpyHostToDevice));

        broadcast(this->buffer_gpu, this->buffer_gpu, size, ncclInt8);
    } else {
        // first recv size
        // [option 1] use zmq to recv
        zmq::message_t msg(sizeof(size));
        this->mq.recv(msg, zmq::recv_flags::none);
        size = *(size_t*) msg.data();
        
        // [option 2] use NcclBroadcast
        // broadcast(this->buffer_gpu.data_ptr(), this->buffer_gpu.data_ptr(), 1, ncclUint64);
        // CUDACHECK(cudaMemcpy(
        //     &size, this->buffer_gpu.data_ptr(), 
        //     sizeof(size_t), 
        //     cudaMemcpyKind::cudaMemcpyDeviceToHost
        // )); // cannot use cudaMemcpyAsync here

        // DMOE_LOG(DEBUG) << "recved size: " << size << LEND;
        ASSERT(size > 0 && size <= GROUP_CHANNEL_BUFFER_SIZE);

        // then recv data
        void* data_buf = (void*) this->buffer_gpu.data_ptr();
        broadcast(this->buffer_gpu, this->buffer_gpu, size, ncclInt8);
        buf = std::malloc(size);
        CUDACHECK(cudaMemcpy(buf, data_buf, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        // DMOE_LOG(DEBUG) << "received metadata " << *decerealize<Metadata>((char*) buf, size) << LEND;
    }
}

void NcclGroupChannel::send_metadata(const Metadata& metadata) {
    tx_range _{"NcclGroupChannel::bcast_metadata"};
    ASSERT(is_root());
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Sending metadata from " << this->local << ":" << metadata << LEND;
    std::string data = cerealize(std::make_shared<Metadata>(metadata));
    void* buf = (void*) data.data();
    size_t size = data.size();
    bcast_obj(buf, size);
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Sent metadata." << LEND;
}

void NcclGroupChannel::recv_metadata(Metadata& metadata) {
    tx_range _{"NcclGroupChannel::recv_metadata"};
    ASSERT(!is_root());
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Recving metadata from " << this->local << LEND;
    void* buf;
    size_t size;
    bcast_obj(buf, size);

    metadata = *decerealize<Metadata>((char*) buf, size);
    std::free(buf);
    // DMOE_LOG(DEBUG) << "NcclGroupChannel Recved metadata." << LEND;
}

void NcclGroupChannel::all_reduce(Tensor tensor, const std::vector<int> &shape) {
    tx_range _{"NcclGroupChannel::all_reduce"};
    void* buf = reinterpret_cast<void*>(tensor.data_ptr());
    int count = 1;
    for (int i: shape)
        count *= i;
    // DMOE_LOG(DEBUG) << "Calling all_reduce for " << count << " elements on stream" << this->stream << LEND;
    NCCLCHECK(ncclAllReduce(
        buf,
        buf,
        count,
        ncclBfloat16,
        ncclSum,
        this->comm,
        this->stream
    ));
    // DMOE_LOG(DEBUG) << "AllReduce done on stream " << this->stream << LEND;
}

Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw) {
    ncclUniqueId& id = *((ncclUniqueId*)(nccl_id_raw));
    auto channel = std::make_shared<NcclChannel>(
        party_local, party_other, id
    );
    // TODO(hogura|20240927): recycle the ncclUniqueId (raw).
    return channel;
}

Channel_t create_zmq_channel(int party_local, int party_other, bool is_sender, int rank) {
    auto channel = std::make_shared<ZmqChannel>(party_local, party_other, is_sender, rank);
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
