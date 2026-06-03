#pragma once

#if USE_NIXL

#include <atomic>

#include "comm.h"
#include "nixl_context.h"

class NixlChannel: public Channel {
public:
    NixlChannel(int party_local, int party_other);
    ~NixlChannel() override;

    void send_raw(uintptr_t data, const BatchMetadata& metadata) override;
    void recv_raw(uintptr_t data, const BatchMetadata& metadata) override;
    void send_batch(const torch::Tensor &data, const BatchMetadata& metadata) override;
    void recv_batch(const torch::Tensor &data, const BatchMetadata& metadata) override;
    void initialize() override {}
    void sync() override {}
    void record_event(cudaEvent_t &event) override {}
    bool is_nixl() const override { return true; }

    int last_slot_id() const { return last_slot_id_; }
    int last_seq() const { return last_seq_; }

private:
    void send_impl(uintptr_t data_ptr, const BatchMetadata& metadata);

    cudaStream_t stream_{nullptr};
    int last_slot_id_{-1};
    int last_seq_{-1};
};

#endif
