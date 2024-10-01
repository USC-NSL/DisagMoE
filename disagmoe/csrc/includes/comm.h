#pragma once

#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"

#include "datatypes.hpp"
#include "cuda_utils.h"

class Channel {
protected:
    int local;
    int other;

    int m_rank() {
        return local < other ? 0 : 1;
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
    NcclChannel(int party_local, int party_other, ncclUniqueId comm_id);

    ~NcclChannel();

    void instantiate() override;

    void send(uintptr_t data, const Metadata& metadata) override;

    void recv(uintptr_t data, const Metadata& metadata) override;
};

Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw);

void* get_nccl_unique_id();

void instantiate_channels(std::vector<Channel_t> channels);