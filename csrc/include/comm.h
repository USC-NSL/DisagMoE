#pragma once

#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "cuda_runtime.h"
#include "zmq.hpp"
#include "nccl.h"
#include "zmq.h"

#include "datatypes.hpp"
#include "cuda_utils.h"

class Channel {
protected:
    int local;
    int other;

    virtual int m_rank() {
        return local < other ? 0 : 1;
    }

    int m_other() {
        return local < other ? 1 : 0;
    }

public:
    Channel(int party_local, int party_other): local(party_local), other(party_other) {}
    ~Channel() {}

    virtual void instantiate() = 0;
    virtual void send(torch::Tensor tensor, const Metadata& metadata) = 0;
    virtual void recv(torch::Tensor tensor, const Metadata& metadata) = 0;

    void _debug_print() {
        printf("%d %d\n", local, other);
    }

    int get_peer_id() const {
        return this->other;
    }

    virtual void sync() {}
};

typedef std::shared_ptr<Channel> Channel_t;

struct cmp_channel_t {
    bool operator()(const Channel_t &l, const Channel_t &r) const {
        return l->get_peer_id() < r->get_peer_id();
    }
};

class NcclChannel: public Channel {
protected:
    ncclUniqueId comm_id;
    ncclComm_t comm;
    cudaStream_t stream;

    void _delay_release_tensor(torch::Tensor tensor, cudaStream_t stream);

public:
    NcclChannel(int party_local, int party_other, ncclUniqueId comm_id, cudaStream_t stream = nullptr);

    ~NcclChannel();

    void instantiate() override;

    void send(torch::Tensor tensor, const Metadata& metadata) override;

    void recv(torch::Tensor tensor, const Metadata& metadata) override;

    void sync() override;
};

typedef std::shared_ptr<zmq::socket_t> mq_t;

class ZmqChannelHelper {
public:
    struct Item;

protected:
    std::thread t;
    bool end_flag;
    mq_t mq;
    cudaStream_t stream;

    std::queue<Item> queue;
    std::mutex mutex, mtx_end;
    std::condition_variable cv, cv_end;

    virtual void run(Item &item) = 0;

public:
    struct Item {
        void* dst {0};
        size_t size {0};
        void* src {0};
        void* event {0};
    };

    ZmqChannelHelper(mq_t mq, cudaStream_t stream);

    void terminate();

    void put(void* dst, size_t size, void* src = nullptr, void* event = 0);

    Item get();

    void start();

    void sync();
};

class ZmqChannelSender: public ZmqChannelHelper {
protected:
    void run(Item &item) override;

public:
    ZmqChannelSender(mq_t mq, cudaStream_t stream): ZmqChannelHelper(mq, stream) {}
};

class ZmqChannelRecver: public ZmqChannelHelper {
protected:
    void run(Item &item) override;

public:
    ZmqChannelRecver(mq_t mq, cudaStream_t stream): ZmqChannelHelper(mq, stream) {}
};

class ZmqChannel: public Channel {

protected:
    static std::map<int, mq_t> global_mq;
    zmq::context_t ctx;
    mq_t mq;
    cudaStream_t stream_send, stream_recv;
    ZmqChannelHelper* helper;

    std::string other_ip;
    bool is_sender;
    char device_id_str[3];

    int rank_offset;

    void* pin_buffer;

public:
    ZmqChannel(int party_local, int party_other, bool is_sender, int rank = 0);
    ~ZmqChannel();

    void instantiate() override;

    void send(torch::Tensor tensor, const Metadata& metadata) override;

    void recv(torch::Tensor tensor, const Metadata &metadata) override;
};

class NcclGroupChannel: public NcclChannel {
protected:
    zmq::context_t ctx;
    zmq::socket_t mq;
    int root_device_id;
    int zmq_comm_id;
    
    int local_rank;
    int size;

    torch::Tensor buffer_gpu;
    int* barrier;

    bool is_root() const;

    int root() const;

    void broadcast(torch::Tensor send_tensor, torch::Tensor recv_tensor, size_t count, ncclDataType_t type, cudaStream_t stream=nullptr);

public:
    NcclGroupChannel(int party_local, const std::vector<int> &party_all, ncclUniqueId comm_id, cudaStream_t stream = nullptr);

    void instantiate() override;

    void send(torch::Tensor tensor, const Metadata& metadata) override;

    void recv(torch::Tensor tensor, const Metadata& metadata) override;

    void synchronize();

    void send_recv(torch::Tensor tensor, const Metadata& metadata);

    void bcast_obj(void* &buf, size_t &size);

    void send_metadata(const Metadata& metadata);

    void recv_metadata(Metadata& metadata);

    void all_reduce(torch::Tensor tensor, const std::vector<int> &shape);
};

Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw);

Channel_t create_zmq_channel(int party_local, int party_other, bool is_sender, int rank = 0);

Channel_t create_nccl_group_channel(int party_local, const std::vector<int> &party_all, void *nccl_id_raw);

std::vector<Channel_t> create_nccl_group_channels(int root, const std::vector<int> &party_all, void *nccl_id_raw);

void* get_nccl_unique_id();

void instantiate_channels(std::vector<Channel_t> channels);