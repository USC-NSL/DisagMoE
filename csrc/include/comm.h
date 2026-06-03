#pragma once

#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include <torch/torch.h>

#include "cuda_runtime.h"
#include "zmq.hpp"
#include "nccl.h"
#include "zmq.h"
#include "ucxq.hpp"

#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
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
    virtual ~Channel() = default;

    virtual void send_raw(uintptr_t data, const BatchMetadata& metadata) = 0;
    virtual void recv_raw(uintptr_t data, const BatchMetadata& metadata) = 0;

    virtual void send_batch(const torch::Tensor &data, const BatchMetadata& metadata) = 0;
    virtual void recv_batch(const torch::Tensor &data, const BatchMetadata& metadata) = 0;

    void _debug_print() {
        printf("%d %d\n", local, other);
    }

    int get_peer_id() const {
        return this->other;
    }

    virtual void initialize() {}

    virtual void sync() {}

    virtual void warmup_send(int *send_buf, int count) {}

    virtual void warmup_recv(int *recv_buf, int count) {}

    virtual void record_event(cudaEvent_t &event) {}

    virtual bool is_nixl() const { return false; }

    virtual bool is_local() const { return false; }

};

typedef std::shared_ptr<Channel> Channel_t;

struct cmp_channel_t {
    bool operator()(const Channel_t &l, const Channel_t &r) const {
        return l->get_peer_id() < r->get_peer_id();
    }
};

class NcclChannel: public Channel {
protected:
    ncclComm_t comm;
    ncclUniqueId unique_id;
    cudaStream_t stream;

public:
    NcclChannel(int party_local, int party_other, ncclUniqueId unique_id, cudaStream_t stream = nullptr);

    void send_raw(uintptr_t data, const BatchMetadata& metadata) override;

    void recv_raw(uintptr_t data, const BatchMetadata& metadata) override;

    void send_batch(const torch::Tensor &data, const BatchMetadata& metadata) override;

    void recv_batch(const torch::Tensor &data, const BatchMetadata& metadata) override;

    void sync() override;

    void initialize() override;

    void warmup_send(int *send_buf, int count) override;

    void warmup_recv(int *recv_buf, int count) override;

    void record_event(cudaEvent_t &event) override;
};

class TensorLocalChannel: public Channel {
    protected:
        cudaStream_t stream;
        std::queue<uintptr_t> data_buffer{};
        std::queue<torch::Tensor> batch_buffer{};
        mutable std::mutex m;
        std::condition_variable c;
    
    public:
        TensorLocalChannel(int device_id, cudaStream_t stream = nullptr);
    
        void send_raw(uintptr_t data, const BatchMetadata& metadata) override;
    
        void recv_raw(uintptr_t data, const BatchMetadata& metadata) override;

        void send_batch(const torch::Tensor &data, const BatchMetadata& metadata) override;

        void recv_batch(const torch::Tensor &data, const BatchMetadata& metadata) override;
    
        void sync() override;

        void record_event(cudaEvent_t &event) override;

        bool is_local() const override { return true; }
};

class NixlChannel;

Channel_t create_nccl_channel(int party_local, int party_other, ncclUniqueId unique_id);

Channel_t create_nixl_channel(int party_local, int party_other);

Channel_t create_local_channel(int device_id);

void* get_nccl_unique_id();
