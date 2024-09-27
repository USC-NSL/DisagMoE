#pragma once

#include <cassert>
#include <memory>

#include "cuda_runtime.h"
#include "nccl.h"

#include "datatypes.h"
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
    virtual void send(void* data, const Metadata& metadata) = 0;
    virtual void recv(void* data, const Metadata& metadata) = 0;
};

typedef std::shared_ptr<Channel> Channel_t;

class NcclChannel: public Channel {
protected:
    ncclUniqueId comm_id;
    ncclComm_t comm;
    cudaStream_t stream;

public:
    NcclChannel(int party_local, int party_other, ncclUniqueId comm_id): Channel::Channel(party_local, party_other), comm_id(comm_id) 
        {
            CUDACHECK(cudaStreamCreate(&this->stream));
        }

    ~NcclChannel() {
        CUDACHECK(cudaStreamDestroy(this->stream));
        delete this->stream;
    }

    void instantiate() override {
        NCCLCHECK(ncclCommInitRank(
            &this->comm,
            /*nranks=*/ 2,
            this->comm_id,
            /*rank=*/ this->m_rank()
        ));
    }

    void send(void* data, const Metadata& metadata) override {
        // TODO(hogura|20240926): check if ncclGroupStart may influence the performance
        NCCLCHECK(ncclSend(
            data, 
            /*count=*/ metadata.get_num_element(),
            /*datatype=*/ metadata.get_nccl_datatype(),
            this->other,
            this->comm,
            this->stream
        ));
    }

    void recv(void* data, const Metadata& metadata) override {
        NCCLCHECK(ncclRecv(
            data,
            /*count=*/ metadata.get_num_element(),
            /*datatype=*/ metadata.get_nccl_datatype(),
            this->other,
            this->comm,
            this->stream
        ));
    }
};