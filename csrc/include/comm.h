#pragma once

#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

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

    virtual void instantiate() = 0;
    virtual void send(uintptr_t data, const Metadata& metadata) = 0;
    virtual void recv(uintptr_t data, const Metadata& metadata) = 0;

    void _debug_print() {
        printf("%d %d\n", local, other);
    }

    int get_peer_id() const {
        return this->other;
    }
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

public:
    NcclChannel(int party_local, int party_other, ncclUniqueId comm_id, cudaStream_t stream = nullptr);

    ~NcclChannel();

    void instantiate() override;

    void send(uintptr_t data, const Metadata& metadata) override;

    void recv(uintptr_t data, const Metadata& metadata) override;
};

typedef std::shared_ptr<zmq::socket_t> mq_t;

class ZmqChannel: public Channel {
protected:
    static std::map<int, mq_t> global_mq;
    zmq::context_t ctx;
    mq_t mq;
    bool is_sender;
    char device_id_str[3];

    void* _tensor_copy(uintptr_t src, const Metadata& metadata, bool to_gpu, uintptr_t dst = 0);

public:
    ZmqChannel(int party_local, int party_other, bool is_sender);

    void instantiate() override;

    void send(uintptr_t data, const Metadata& metadata) override;

    void recv(uintptr_t data, const Metadata &metadata) override;
};

class NcclGroupChannel: public NcclChannel {
protected:
    zmq::context_t ctx;
    zmq::socket_t mq;
    int root_device_id;
    int zmq_comm_id;
    
    int local_rank;
    int size;

    int* barrier;

    bool is_root() const;

    int root() const;

    void broadcast(void* send_buf, void* recv_buf, size_t count, ncclDataType_t type, cudaStream_t stream=nullptr);

public:
    NcclGroupChannel(int party_local, const std::vector<int> &party_all, ncclUniqueId comm_id, cudaStream_t stream = nullptr);

    void instantiate() override;

    void send(uintptr_t data, const Metadata& metadata) override;

    void recv(uintptr_t data, const Metadata& metadata) override;

    void synchronize();

    void send_recv(uintptr_t data, const Metadata& metadata);

    void bcast_obj(void* &buf, size_t &size, cudaStream_t stream = nullptr);

    void send_metadata(const Metadata& metadata);

    void recv_metadata(Metadata& metadata);

    void all_reduce(uintptr_t data, const std::vector<int> &shape);
};

Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw);

Channel_t create_zmq_channel(int party_local, int party_other, bool is_sender);

Channel_t create_nccl_group_channel(int party_local, const std::vector<int> &party_all, void *nccl_id_raw);

std::vector<Channel_t> create_nccl_group_channels(int root, const std::vector<int> &party_all, void *nccl_id_raw);

void* get_nccl_unique_id();

void instantiate_channels(std::vector<Channel_t> channels);