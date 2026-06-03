#pragma once

#if USE_NIXL

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cuda_runtime.h"
#include "metadata.hpp"
#include "wrapper.h"

typedef void nixlXferReqH;

constexpr int NIXL_SEND_RING_SIZE = 32;
constexpr int NIXL_RECV_RING_SIZE = 8;
constexpr size_t NIXL_MAX_BATCH_BYTES = 4ULL * 1024 * 1024;

// [V2] Binary notif format. 16-byte header followed by `meta_len` bytes of
// cerealized BatchMetadata. type='D' carries metadata after the data WRITE;
// type='C' is a credit return (header-only, meta_len=0, slot_id=consumed
// recv slot). The seq field is set on 'D' notifs only.
#pragma pack(push, 1)
struct NixlNotifHeader {
    uint8_t  type;
    uint8_t  reserved[3];
    uint32_t seq;
    uint32_t slot_id;
    uint32_t meta_len;
};
#pragma pack(pop)
static_assert(sizeof(NixlNotifHeader) == 16, "NixlNotifHeader must be 16 bytes");
constexpr uint8_t NIXL_NOTIF_DATA = 'D';
constexpr uint8_t NIXL_NOTIF_CREDIT = 'C';

// [TRACING] Per-call subtimings populated by NixlContext::post_write when
// the caller passes a non-null pointer. All values are wall-clock seconds.
struct NixlPostWriteTimings {
    double dt_create_s{0.0};   // make_xfer_req + post_xfer_req combined
    double dt_make_s{0.0};     // make_xfer_req only
    double dt_post_s{0.0};     // post_xfer_req only
    double dt_other_s{0.0};    // opts setup/cleanup
};

// [TRACING-MAY4] Send-trace tuple field layout (positional, must match
// Python parser nixl_send_trace_keys() in advanced_logger.py):
//   0  peer_id
//   1  seq
//   2  recv_slot
//   3  bytes
//   4  t_enter_send_s          (t0)  entry to send_impl
//   5  dt_acq_recv_s           (t1-t0) acquire_recv_slot (credit wait)
//   6  dt_acq_send_s           (t2-t1) acquire_send_slot
//   7  dt_d2d_enqueue_s        (t3-t2) cudaMemcpyAsync + EventRecord
//   8  dt_build_notif_s        (t4-t3) cereal serialize metadata
//   9  dt_dispatcher_total_s   (t5-t0) total dispatcher path
//  10  dt_queue_wait_s         (t6-t5) dispatcher enqueue -> post_worker pickup
//  11  dt_d2d_busy_poll_s      (t7-t6) busy-poll cudaEventQuery
//  12  dt_post_create_s              make_xfer_req + post_xfer_req
//  13  dt_post_make_s                make_xfer_req only
//  14  dt_post_xfer_s                post_xfer_req only
//  15  dt_post_other_s               opt_args setup + cleanup
//  16  dt_total_s              (t8-t0) end-to-end send wall time
using NixlSendTraceTuple = std::tuple<
    int, int, int, size_t,
    double,
    double, double, double, double,
    double,
    double, double,
    double, double, double,
    double,
    double>;

// [TRACING-MAY4] Recv-trace tuple field layout (positional, must match
// Python parser nixl_recv_trace_keys() in advanced_logger.py):
//   0  peer_id
//   1  seq
//   2  slot_id
//   3  bytes
//   4  t_arrived_s             (tR0) poll_worker drained notif
//   5  dt_pool_pickup_s        (tR3-tR0) drained -> MuPool consumed it
//   6  dt_alloc_d2d_issue_s    (tR4-tR3) torch::empty + cudaMemcpyAsync + EventRecord
//   7  dt_d2d_complete_s       (tR5-tR4) cudaEventQuery loop until ready
//   8  dt_credit_send_s        (tR6-tR5) send_credit (gen_notif)
//   9  dt_total_s              (tR6-tR0) notif arrival -> credit returned
using NixlRecvTraceTuple = std::tuple<
    int, int, int, size_t,
    double,
    double, double,
    double, double,
    double>;

// [V2] Result of a 'D' notif: the receiver pulls these out of the NIXL
// context after each poll_notifs() and feeds them into pending_metas_.
struct NixlMetaArrival {
    int peer_id;
    int seq;
    int slot_id;
    batch_metadata_t meta;
    double t_arrived_s{0.0};
};

struct NixlPendingPost {
    int peer_id;
    int send_slot;
    int recv_slot;
    int seq;
    size_t bytes_to_write;
    std::string notif_payload;
    cudaEvent_t d2d_event;
    double t_enqueued_s{0.0};
    bool tracing{false};
    double t_enter_send_s{0.0};
    double dt_acq_recv_s{0.0};
    double dt_acq_send_s{0.0};
    double dt_d2d_enqueue_s{0.0};
    double dt_build_notif_s{0.0};
    double dt_dispatcher_total_s{0.0};
};

class NixlContext {
public:
    static NixlContext& instance();

    void initialize(int local_global_id, const std::vector<int>& peer_global_ids, int local_device_id);
    void shutdown();

    int acquire_send_slot();
    void release_send_slot(int slot_id);
    uintptr_t send_slot_ptr(int slot_id) const;
    size_t slot_size() const;

    int acquire_recv_slot(int peer_global_id);

    uintptr_t recv_slot_ptr(int peer_global_id, int slot_id) const;

    nixlXferReqH* post_write(int peer_global_id, int send_slot, int recv_slot,
                             size_t bytes_to_write, const std::string& notif,
                             NixlPostWriteTimings* out_timings,
                             nixl_capi_xfer_dlist_t* out_pc_local,
                             nixl_capi_xfer_dlist_t* out_pc_remote);

    int xfer_status(nixlXferReqH* h);
    void release_xfer(nixlXferReqH* h);

    void enqueue_pending_xfer(nixlXferReqH* handle, int send_slot, std::string notif_keepalive,
                              int peer_id, int recv_slot, int seq,
                              nixl_capi_xfer_dlist_t pc_local,
                              nixl_capi_xfer_dlist_t pc_remote);
    bool has_pending_xfers() const;
    int drain_pending_xfers();

    // [V2-B] The dispatcher hands a PendingPost to the post_worker thread,
    // which calls cudaEventSynchronize on the D2D event and then post_write.
    void enqueue_post(NixlPendingPost&& post);

    // [V2-A] poll_notifs parses binary notifs. 'D' notifs become
    // NixlMetaArrival entries available via drain_meta_arrivals(); 'C' notifs
    // call on_recv_credit() directly.
    void poll_notifs();
    std::vector<NixlMetaArrival> drain_meta_arrivals();
    bool has_meta_arrivals() const { return arrival_count_.load(std::memory_order_relaxed) > 0; }

    // [V4] NIXL notif semantics (per official NIXL BackendGuide.md, UCX docs):
    //   * RELIABLE delivery on IB-RC / RoCE(PFC) / TCP — messages do NOT silently
    //     drop on healthy links; failure is reported via explicit error codes.
    //   * NO ordering guarantee across distinct transfer requests; standalone
    //     genNotif() also has no ordering guarantee.
    //   * Visibility latency at the receiver = drain rate of the NIXL/UCX
    //     user-space notif queue. We solve this with a dedicated poll thread
    //     (see poll_worker_loop) that calls get_notifs() continuously, so the
    //     queue is drained in microseconds. This obviates any retransmit /
    //     resync mechanism — RDMA + a fast poller is sufficient.
    //
    // Sender slot ownership is identified by (peer, slot, seq), not (peer, slot)
    // alone, because a slot is reused across many sends and a delayed/duplicate
    // C(slot) from an old use must not ack a current use. The seq tag is the
    // ONLY thing protecting against reordered credits; there is NO retransmit.
    void send_credit(int peer_global_id, int slot_id, int seq);
    int next_send_seq(int peer_global_id);
    bool accept_data_notif(int peer_global_id, int slot_id, int seq);
    void mark_recv_completed(int peer_global_id, int slot_id, int seq);

    int local_id() const { return local_global_id_; }
    bool is_initialized() const { return initialized_; }
    cudaStream_t recv_stream() const { return recv_stream_; }

    void set_tracing_enabled(bool v) { tracing_enabled_.store(v, std::memory_order_relaxed); }
    bool tracing_enabled() const { return tracing_enabled_.load(std::memory_order_relaxed); }
    void record_send_trace(const NixlSendTraceTuple& trace);
    void record_recv_trace(const NixlRecvTraceTuple& trace);
    std::vector<NixlSendTraceTuple> drain_send_traces();
    std::vector<NixlRecvTraceTuple> drain_recv_traces();

    void note_send_posted(int peer_global_id, int slot_id, int seq);
    void note_credit_sent(int peer_global_id);
    std::string dump_diagnostic() const;

private:
    struct PeerState {
        int peer_global_id{-1};
        std::string peer_name;
        int peer_dev_id{0};
        void* recv_buf{nullptr};
        nixl_capi_reg_dlist_t recv_reg_dlist{nullptr};
        /* nixl_warmup_dlist + nixl_warmup_dlist_h: long-lived NIXL prepared
         * descriptor list for this peer's recv slots. NOT used by post_write
         * (which builds per-call raw dlists for create_xfer_req). They exist
         * solely because NIXL's per-worker rkey unpack (ucp_ep_rkey_unpack)
         * is cached inside the prepared handle's metadata; releasing the
         * handle drops that cache and every subsequent create_xfer_req call
         * pays the unpack cost, causing a ~20x per-token-ITL regression.
         * Keep alive for the engine's lifetime. */
        nixl_capi_xfer_dlist_t nixl_warmup_dlist{nullptr};
        nixl_capi_xfer_dlist_handle_t nixl_warmup_dlist_h{nullptr};
        struct SlotDesc {
            uintptr_t addr;
            size_t len;
            uint64_t dev_id;
        };
        std::vector<SlotDesc> recv_slot_descs;
        std::vector<SlotDesc> remote_recv_slot_descs;
        std::atomic<int> credits{0};
        std::deque<int> available_recv_slots;
        mutable std::mutex credit_mutex;
        std::condition_variable credit_cv;
        std::atomic<uint64_t> sends_posted{0};
        std::atomic<uint64_t> credits_received{0};
        std::atomic<uint64_t> credits_sent{0};

        std::atomic<int> next_seq{0};

        // [V4] Slot ownership tag. Sender uses expected_credit_seq to validate
        // an incoming credit (rejects late credits from previous slot uses);
        // receiver uses inflight_recv_seq / completed_recv_seq to dedupe
        // reordered/duplicate D-notifs. No retransmit fields — RDMA is reliable
        // and the dedicated poll thread drains notifs in microseconds.
        struct SlotState {
            int expected_credit_seq{-1};
            int inflight_recv_seq{-1};
            int completed_recv_seq{-1};
        };
        std::vector<SlotState> slot_state;
    };

    NixlContext() = default;
    ~NixlContext() = default;
    NixlContext(const NixlContext&) = delete;
    NixlContext& operator=(const NixlContext&) = delete;

    PeerState& peer_state(int peer_global_id);
    const PeerState& peer_state(int peer_global_id) const;
    void check_nixl_status(nixl_capi_status_t status, const std::string& what) const;
    void exchange_remote_metadata(const std::vector<int>& peer_global_ids);
    void on_recv_credit(int peer_global_id, int slot_id, int seq);

    mutable std::mutex init_mutex_;
    bool initialized_{false};
    int local_global_id_{-1};
    int local_device_id_{0};
    std::string local_agent_name_;
    nixl_capi_agent_t agent_{nullptr};
    nixl_capi_backend_t backend_{nullptr};
    nixl_capi_opt_args_t backend_opts_{nullptr};

    // [V2-C] Cached opt_args used by post_write inside the post_worker.
    // Only the post_worker mutates notif_msg on it, so no external lock
    // is required.
    nixl_capi_opt_args_t cached_post_opts_{nullptr};

    void* send_buf_{nullptr};
    nixl_capi_reg_dlist_t send_reg_dlist_{nullptr};
    /* See PeerState::nixl_warmup_dlist comment. The send-side warmup dlist
     * is the local-side analog: NIXL's prepared-handle metadata caches the
     * MR lookups for these source addresses. Kept alive for the engine's
     * lifetime. */
    nixl_capi_xfer_dlist_t send_warmup_dlist_{nullptr};
    nixl_capi_xfer_dlist_handle_t send_warmup_dlist_h_{nullptr};
    cudaStream_t recv_stream_{nullptr};

    std::unordered_map<int, std::unique_ptr<PeerState>> peers_;

    std::vector<bool> send_slot_busy_;
    int next_send_slot_{0};
    mutable std::mutex send_mutex_;
    std::condition_variable send_cv_;

    // [V2-A] poll_notifs parses 'D' notifs and stages NixlMetaArrival
    // entries here for the pool loop to drain. Replaces the old ready_set_
    // (R<seq> string parsing) and is the only path metadata reaches the
    // receiver in V2.
    mutable std::mutex notif_mutex_;
    std::vector<NixlMetaArrival> pending_arrivals_;
    std::atomic<int> arrival_count_{0};

    std::atomic<bool> tracing_enabled_{false};
    mutable std::mutex send_traces_mutex_;
    mutable std::mutex recv_traces_mutex_;
    std::vector<NixlSendTraceTuple> send_traces_;
    std::vector<NixlRecvTraceTuple> recv_traces_;

    struct PendingXfer {
        nixlXferReqH* handle;
        int send_slot;
        std::string notif_keepalive;
        int peer_id{-1};
        int recv_slot{-1};
        int seq{-1};
        nixl_capi_xfer_dlist_t pc_local{nullptr};
        nixl_capi_xfer_dlist_t pc_remote{nullptr};
    };
    mutable std::mutex pending_xfers_mutex_;
    std::deque<PendingXfer> pending_xfers_;

    std::deque<NixlPendingPost> post_queue_;
    mutable std::mutex post_mutex_;
    std::condition_variable post_cv_;
    /* DESIGN_A0+B3: multiple post_worker threads. Each thread is assigned its
     * own ucp_worker via NIXL's thread_local getWorkerId() cache, enabling
     * true parallel posting across the 4 ucp_workers configured by A0. */
    std::vector<std::thread> post_workers_;
    std::atomic<bool> post_worker_stop_{false};

    void post_worker_loop();

    // [V4] Dedicated NIXL notification poll thread. Spins on get_notifs() so
    // notifs are drained from the NIXL/UCX user-space queue with sub-millisecond
    // latency, regardless of what the pool/dispatcher threads are doing.
    // Replaces the previous "every-128-iters" throttle inside MuPool::run, and
    // makes the V3 5-second resync watchdog unnecessary (and removed).
    std::thread poll_worker_;
    std::atomic<bool> poll_worker_stop_{false};
    void poll_worker_loop();
};

#endif
