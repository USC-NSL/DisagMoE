#pragma once

#include <stdexcept>
#include <cassert>
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

    int m_rank() {
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

Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw);
Channel_t create_zmq_channel(int party_local, int party_other, bool is_sender);

void* get_nccl_unique_id();

void instantiate_channels(std::vector<Channel_t> channels);