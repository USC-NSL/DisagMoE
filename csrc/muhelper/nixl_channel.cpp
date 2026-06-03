#include "nixl_channel.h"

#if USE_NIXL

#include <chrono>
#include <cstring>
#include <string>

#include <cereal/archives/binary.hpp>

#include "logging.h"
#include "metadata.hpp"
#include "utils.hpp"

namespace {
inline double nixl_wall_time_s() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// [V2-A] Build the binary 'D' notif: 16-byte header + cerealized metadata.
inline std::string build_data_notif(int recv_slot, int seq, const BatchMetadata& meta) {
    std::string meta_blob = cerealize_(meta);
    std::string out;
    out.resize(sizeof(NixlNotifHeader) + meta_blob.size());
    NixlNotifHeader hdr{};
    hdr.type = NIXL_NOTIF_DATA;
    hdr.seq = static_cast<uint32_t>(seq);
    hdr.slot_id = static_cast<uint32_t>(recv_slot);
    hdr.meta_len = static_cast<uint32_t>(meta_blob.size());
    std::memcpy(out.data(), &hdr, sizeof(hdr));
    if (!meta_blob.empty()) {
        std::memcpy(out.data() + sizeof(hdr), meta_blob.data(), meta_blob.size());
    }
    return out;
}
}

NixlChannel::NixlChannel(int party_local, int party_other)
    : Channel(party_local, party_other) {
#ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->local));
#endif
    CUDACHECK(cudaStreamCreate(&stream_));
}

NixlChannel::~NixlChannel() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void NixlChannel::send_raw(uintptr_t data, const BatchMetadata& metadata) {
    send_impl(data, metadata);
}

void NixlChannel::recv_raw(uintptr_t, const BatchMetadata&) {
    ASSERT_MSG(false, "NixlChannel::recv_raw should not be used");
}

void NixlChannel::send_batch(const torch::Tensor &data, const BatchMetadata& metadata) {
    send_impl(reinterpret_cast<uintptr_t>(data.data_ptr()), metadata);
}

void NixlChannel::recv_batch(const torch::Tensor &, const BatchMetadata&) {
    ASSERT_MSG(false, "NixlChannel::recv_batch should not be used");
}

// [V2-A][V2-B] send_impl no longer blocks on cudaStreamSynchronize. Instead
// it records a CUDA event after the staging D2D copy and hands off a
// PendingPost to NixlContext::post_worker. The notif payload (V2-A binary
// header + cerealized metadata) is built here so the post_worker only has
// to issue the RDMA WRITE.
void NixlChannel::send_impl(uintptr_t data_ptr, const BatchMetadata& metadata) {
    if (metadata.num_element() == 0) {
        last_slot_id_ = -1;
        last_seq_ = -1;
        return;
    }

    auto& ctx = NixlContext::instance();
    const bool tracing = ctx.tracing_enabled();
    const double t0 = tracing ? nixl_wall_time_s() : 0.0;

    const int recv_slot = ctx.acquire_recv_slot(this->other);
    const double t1 = tracing ? nixl_wall_time_s() : 0.0;
    const int send_slot = ctx.acquire_send_slot();
    const double t2 = tracing ? nixl_wall_time_s() : 0.0;
    const size_t bytes_to_write = metadata.num_element() * metadata.get_datatype_size();

    CUDACHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(ctx.send_slot_ptr(send_slot)),
        reinterpret_cast<void*>(data_ptr),
        bytes_to_write,
        cudaMemcpyDeviceToDevice,
        stream_));

    cudaEvent_t d2d_event;
    CUDACHECK(cudaEventCreateWithFlags(&d2d_event, cudaEventDisableTiming));
    CUDACHECK(cudaEventRecord(d2d_event, stream_));

    const double t3 = tracing ? nixl_wall_time_s() : 0.0;

    const int seq = ctx.next_send_seq(this->other);
    auto notif_payload = build_data_notif(recv_slot, seq, metadata);
    const double t4 = tracing ? nixl_wall_time_s() : 0.0;

    NixlPendingPost post;
    post.peer_id = this->other;
    post.send_slot = send_slot;
    post.recv_slot = recv_slot;
    post.seq = seq;
    post.bytes_to_write = bytes_to_write;
    post.notif_payload = std::move(notif_payload);
    post.d2d_event = d2d_event;
    post.tracing = tracing;
    if (tracing) {
        post.t_enter_send_s = t0;
        post.dt_acq_recv_s = t1 - t0;
        post.dt_acq_send_s = t2 - t1;
        post.dt_d2d_enqueue_s = t3 - t2;
        post.dt_build_notif_s = t4 - t3;
    }

    ctx.note_send_posted(this->other, recv_slot, seq);

    const double t5 = tracing ? nixl_wall_time_s() : 0.0;
    if (tracing) {
        post.t_enqueued_s = t5;
        post.dt_dispatcher_total_s = t5 - t0;
    }
    ctx.enqueue_post(std::move(post));

    last_slot_id_ = recv_slot;
    last_seq_ = seq;
}

#endif
