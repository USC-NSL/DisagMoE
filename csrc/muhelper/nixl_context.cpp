#include "nixl_context.h"

#if USE_NIXL

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "constants.h"
#include "cuda_utils.h"
#include "distributed.hpp"
#include "logging.h"
#include "utils.hpp"
#include "zmq.hpp"

namespace {

inline double nixl_wall_time_s() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

constexpr int ZMQ_NIXL_INIT_PORT_BASE = 61000;
constexpr int ZMQ_NIXL_DESC_PORT_BASE = 61300;
constexpr int ZMQ_NIXL_READY_PORT_BASE = 61600;
constexpr int ZMQ_TIMEOUT_MS = 1800000;  // 30 min - generous, multi-node model load can be slow

struct NixlExchangeEnvelope {
    int sender_id;
    std::string payload;

    template<class Archive>
    void serialize(Archive& archive) {
        archive(sender_id, payload);
    }
};

struct NixlDescriptorEnvelope {
    int peer_dev_id;
    struct DescTriple {
        uintptr_t addr;
        size_t len;
        uint64_t dev_id;

        template<class Archive>
        void serialize(Archive& archive) {
            archive(addr, len, dev_id);
        }
    };
    std::vector<DescTriple> descs;

    template<class Archive>
    void serialize(Archive& archive) {
        archive(peer_dev_id, descs);
    }
};

inline std::string make_agent_name(int rank) {
    return "rank_" + std::to_string(rank);
}

inline int parse_agent_rank(const std::string& agent_name) {
    const std::string prefix = "rank_";
    if (agent_name.rfind(prefix, 0) != 0) {
        return -1;
    }
    try {
        return std::stoi(agent_name.substr(prefix.size()));
    } catch (...) {
        return -1;
    }
}

inline std::string endpoint_for(int rank, int base_port) {
    return get_zmq_addr(rank, true, base_port, 0);
}

inline std::string bind_endpoint_for(int rank, int base_port) {
    const auto endpoint = endpoint_for(rank, base_port);
    const auto pos = endpoint.rfind(':');
    ASSERT_MSG(pos != std::string::npos, "Invalid ZMQ endpoint");
    return "tcp://*:" + endpoint.substr(pos + 1);
}

std::unordered_map<int, std::string> exchange_phase(
    int local_rank,
    const std::vector<int>& peers,
    int base_port,
    const std::unordered_map<int, std::string>& payloads) {
    static zmq::context_t ctx(1);
    zmq::socket_t pull(ctx, zmq::socket_type::pull);
    pull.set(zmq::sockopt::linger, ZMQ_TIMEOUT_MS);
    pull.set(zmq::sockopt::rcvtimeo, ZMQ_TIMEOUT_MS);
    pull.set(zmq::sockopt::rcvhwm, 0);
    pull.bind(bind_endpoint_for(local_rank, base_port));

    DMOE_LOG(INFO) << "NIXL exchange phase start: rank=" << local_rank << ", port_base=" << base_port << ", peers=" << peers.size() << LEND;

    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::vector<std::unique_ptr<zmq::socket_t>> pushes;
    pushes.reserve(peers.size());
    for (int peer_id : peers) {
        auto payload_it = payloads.find(peer_id);
        ASSERT_MSG(payload_it != payloads.end(), "Missing NIXL exchange payload for peer");
        const auto serialized = cerealize_(NixlExchangeEnvelope{local_rank, payload_it->second});
        auto sock = std::make_unique<zmq::socket_t>(ctx, zmq::socket_type::push);
        sock->set(zmq::sockopt::linger, ZMQ_TIMEOUT_MS);
        sock->set(zmq::sockopt::sndhwm, 0);
        sock->set(zmq::sockopt::sndtimeo, ZMQ_TIMEOUT_MS);
        sock->set(zmq::sockopt::reconnect_ivl, 100);
        sock->connect(endpoint_for(peer_id, base_port));
        sock->send(zmq::buffer(serialized.data(), serialized.size()), zmq::send_flags::none);
        pushes.emplace_back(std::move(sock));
    }
    DMOE_LOG(INFO) << "NIXL exchange phase: rank=" << local_rank << " all sends queued (n=" << peers.size() << ")" << LEND;

    std::unordered_map<int, std::string> out;
    while (out.size() < peers.size()) {
        zmq::message_t msg;
        auto recv_res = pull.recv(msg, zmq::recv_flags::none);
        ASSERT_MSG(recv_res.has_value(), "Timed out during NIXL metadata exchange");
        NixlExchangeEnvelope env;
        decerealize_(reinterpret_cast<char*>(msg.data()), msg.size(), env);
        out[env.sender_id] = std::move(env.payload);
        DMOE_LOG(INFO) << "NIXL exchange recv: rank=" << local_rank << " <- peer=" << env.sender_id << ", port_base=" << base_port << ", progress=" << out.size() << "/" << peers.size() << LEND;
    }
    DMOE_LOG(INFO) << "NIXL exchange phase done: rank=" << local_rank << ", port_base=" << base_port << LEND;
    return out;
}

}  // namespace

NixlContext& NixlContext::instance() {
    static NixlContext ctx;
    return ctx;
}

void NixlContext::check_nixl_status(nixl_capi_status_t status, const std::string& what) const {
    if (status != NIXL_CAPI_SUCCESS && status != NIXL_CAPI_IN_PROG) {
        DMOE_LOG(ERROR) << what << " failed with status " << status << LEND;
        throw std::runtime_error(what + " failed with status " + std::to_string(status));
    }
}

NixlContext::PeerState& NixlContext::peer_state(int peer_global_id) {
    auto it = peers_.find(peer_global_id);
    ASSERT_MSG(it != peers_.end(), "Unknown NIXL peer id");
    return *it->second;
}

const NixlContext::PeerState& NixlContext::peer_state(int peer_global_id) const {
    auto it = peers_.find(peer_global_id);
    ASSERT_MSG(it != peers_.end(), "Unknown NIXL peer id");
    return *it->second;
}

void NixlContext::initialize(int local_global_id, const std::vector<int>& peer_global_ids, int local_device_id) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initialized_) {
        return;
    }

    local_global_id_ = local_global_id;
    local_device_id_ = local_device_id;
    local_agent_name_ = make_agent_name(local_global_id_);

    std::vector<int> sorted_peers = peer_global_ids;
    std::sort(sorted_peers.begin(), sorted_peers.end());
    sorted_peers.erase(std::unique(sorted_peers.begin(), sorted_peers.end()), sorted_peers.end());
    DMOE_LOG(INFO) << "NIXL initialize begin: rank=" << local_global_id_ << ", peers=" << sorted_peers.size() << LEND;

    /* DESIGN_A0 fix #1: explicit RW thread_sync (avoid STRICT auto-upgrade
     * triggered by useListenThread=true; STRICT serializes all postXferReq
     * via the agent's NIXL_SHARED_LOCK_GUARD). */
    nixl_capi_agent_config_t cfg{};
    cfg.enable_prog_thread = true;
    cfg.enable_listen_thread = true;
    cfg.listen_port = 60000 + local_global_id_;
    cfg.lthr_delay_us = 100000;
    cfg.pthr_delay_us = 0;
    cfg.thread_sync = NIXL_CAPI_THREAD_SYNC_RW;
    cfg.num_workers = 4;
    check_nixl_status(nixl_capi_create_configured_agent(local_agent_name_.c_str(), &cfg, &agent_), "create_configured_agent");

    CUDACHECK(cudaSetDevice(local_device_id_));
    CUDACHECK(cudaStreamCreateWithFlags(&recv_stream_, cudaStreamNonBlocking));

    nixl_capi_string_list_t plugins = nullptr;
    check_nixl_status(nixl_capi_get_available_plugins(agent_, &plugins), "get_available_plugins");
    size_t plugin_count = 0;
    check_nixl_status(nixl_capi_string_list_size(plugins, &plugin_count), "string_list_size");
    bool found_ucx = false;
    for (size_t i = 0; i < plugin_count; ++i) {
        const char* plugin_name = nullptr;
        check_nixl_status(nixl_capi_string_list_get(plugins, i, &plugin_name), "string_list_get");
        if (std::string(plugin_name) == "UCX") {
            found_ucx = true;
            break;
        }
    }
    nixl_capi_destroy_string_list(plugins);
    ASSERT_MSG(found_ucx, "UCX backend not available in NIXL");

    nixl_capi_mem_list_t mems = nullptr;
    nixl_capi_params_t params = nullptr;
    check_nixl_status(nixl_capi_get_plugin_params(agent_, "UCX", &mems, &params), "get_plugin_params");
    /* DESIGN_A0 fix #2: explicitly set UCX backend num_workers (default = 1).
     * The agent-level cfg.num_workers field is silently dropped by the C API
     * wrapper, so we must set it via plugin params here. Combined with
     * thread_local worker assignment in nixlUcxEngine::getWorkerId, this
     * lets concurrent post threads use distinct ucp_workers. */
    check_nixl_status(nixl_capi_params_add(params, "num_workers", "4"), "params_add(num_workers)");
    check_nixl_status(nixl_capi_create_backend(agent_, "UCX", params, &backend_), "create_backend");
    nixl_capi_destroy_mem_list(mems);
    nixl_capi_destroy_params(params);

    check_nixl_status(nixl_capi_create_opt_args(&backend_opts_), "create_opt_args");
    check_nixl_status(nixl_capi_opt_args_add_backend(backend_opts_, backend_), "opt_args_add_backend");

    check_nixl_status(nixl_capi_create_opt_args(&cached_post_opts_), "create_opt_args(cached_post)");
    check_nixl_status(nixl_capi_opt_args_add_backend(cached_post_opts_, backend_), "opt_args_add_backend(cached_post)");
    check_nixl_status(nixl_capi_opt_args_set_has_notif(cached_post_opts_, true), "opt_args_set_has_notif(cached_post)");

    CUDACHECK(cudaMalloc(&send_buf_, NIXL_SEND_RING_SIZE * NIXL_MAX_BATCH_BYTES));
    check_nixl_status(nixl_capi_create_reg_dlist(NIXL_CAPI_MEM_VRAM, &send_reg_dlist_), "create_reg_dlist(send)");
    check_nixl_status(nixl_capi_reg_dlist_add_desc(
        send_reg_dlist_, reinterpret_cast<uintptr_t>(send_buf_), NIXL_SEND_RING_SIZE * NIXL_MAX_BATCH_BYTES, local_device_id_, nullptr, 0),
        "reg_dlist_add_desc(send)");
    check_nixl_status(nixl_capi_register_mem(agent_, send_reg_dlist_, backend_opts_), "register_mem(send)");

    /* See send_warmup_dlist_ comment in nixl_context.h. The prepared handle
     * is held for the engine's lifetime to keep NIXL's local-MR metadata
     * cache populated; post_write doesn't reference these handles, but
     * destroying them collapses NIXL's per-call setup path and regresses
     * per-token ITL ~20x. */
    check_nixl_status(nixl_capi_create_xfer_dlist(NIXL_CAPI_MEM_VRAM, &send_warmup_dlist_), "create_xfer_dlist(send-warmup)");
    for (int i = 0; i < NIXL_SEND_RING_SIZE; ++i) {
        check_nixl_status(nixl_capi_xfer_dlist_add_desc(
            send_warmup_dlist_, reinterpret_cast<uintptr_t>(send_buf_) + i * NIXL_MAX_BATCH_BYTES,
            NIXL_MAX_BATCH_BYTES, local_device_id_), "xfer_dlist_add_desc(send-warmup)");
    }
    check_nixl_status(nixl_capi_prep_xfer_dlist(agent_, "", send_warmup_dlist_, &send_warmup_dlist_h_, backend_opts_),
                      "prep_xfer_dlist(send-warmup)");

    send_slot_busy_.assign(NIXL_SEND_RING_SIZE, false);

    for (int peer_id : sorted_peers) {
        auto [it, inserted] = peers_.emplace(peer_id, std::make_unique<PeerState>());
        (void)inserted;
        auto& peer = *it->second;
        peer.peer_global_id = peer_id;
        peer.peer_name = make_agent_name(peer_id);
        peer.credits.store(NIXL_RECV_RING_SIZE);
        for (int slot = 0; slot < NIXL_RECV_RING_SIZE; ++slot) {
            peer.available_recv_slots.push_back(slot);
        }

        peer.slot_state.resize(NIXL_RECV_RING_SIZE);

        CUDACHECK(cudaMalloc(&peer.recv_buf, NIXL_RECV_RING_SIZE * NIXL_MAX_BATCH_BYTES));
        check_nixl_status(nixl_capi_create_reg_dlist(NIXL_CAPI_MEM_VRAM, &peer.recv_reg_dlist), "create_reg_dlist(recv)");
        check_nixl_status(nixl_capi_reg_dlist_add_desc(
            peer.recv_reg_dlist, reinterpret_cast<uintptr_t>(peer.recv_buf), NIXL_RECV_RING_SIZE * NIXL_MAX_BATCH_BYTES, local_device_id_, nullptr, 0),
            "reg_dlist_add_desc(recv)");
        check_nixl_status(nixl_capi_register_mem(agent_, peer.recv_reg_dlist, backend_opts_), "register_mem(recv)");

        for (int slot = 0; slot < NIXL_RECV_RING_SIZE; ++slot) {
            peer.recv_slot_descs.push_back(PeerState::SlotDesc{
                reinterpret_cast<uintptr_t>(peer.recv_buf) + slot * NIXL_MAX_BATCH_BYTES,
                NIXL_MAX_BATCH_BYTES,
                static_cast<uint64_t>(local_device_id_),
            });
        }
    }

    exchange_remote_metadata(sorted_peers);

    post_worker_stop_.store(false);
    /* DESIGN_A0+B3: spawn 4 post_worker threads to match num_workers=4 in
     * UCX backend. NIXL's getWorkerId() uses static thread_local mapping,
     * so each thread will be assigned a distinct ucp_worker on first call. */
    constexpr int kNumPostWorkers = 4;
    for (int i = 0; i < kNumPostWorkers; ++i) {
        post_workers_.emplace_back([this, i]() {
            CUDACHECK(cudaSetDevice(local_device_id_));
            std::string th_name = "NixlPostWorker" + std::to_string(i) + "@" + std::to_string(local_global_id_);
            pthread_setname_np(pthread_self(), th_name.c_str());
            post_worker_loop();
        });
    }

    // [V4] Dedicated NIXL notif polling thread. Drains get_notifs() at
    // ~user-space rate so D-notifs become visible to MuPool and C-notifs
    // unblock the dispatcher within microseconds, instead of the milliseconds-
    // to-seconds the in-loop throttled polling produced.
    poll_worker_stop_.store(false);
    poll_worker_ = std::thread([this]() {
        std::string th_name = "NixlPollWorker@" + std::to_string(local_global_id_);
        pthread_setname_np(pthread_self(), th_name.c_str());
        poll_worker_loop();
    });

    initialized_ = true;
    DMOE_LOG(INFO) << "NIXL initialize done: rank=" << local_global_id_ << LEND;
}

void NixlContext::exchange_remote_metadata(const std::vector<int>& peer_global_ids) {
    void* md_data = nullptr;
    size_t md_len = 0;
    check_nixl_status(nixl_capi_get_local_md(agent_, &md_data, &md_len), "get_local_md");
    std::string local_md(reinterpret_cast<char*>(md_data), md_len);
    std::free(md_data);

    std::unordered_map<int, std::string> md_payloads;
    for (int peer_id : peer_global_ids) {
        md_payloads.emplace(peer_id, local_md);
    }
    auto remote_mds = exchange_phase(local_global_id_, peer_global_ids, ZMQ_NIXL_INIT_PORT_BASE, md_payloads);
    for (auto& [peer_id, remote_md] : remote_mds) {
        char* agent_name = nullptr;
        check_nixl_status(nixl_capi_load_remote_md(agent_, remote_md.data(), remote_md.size(), &agent_name), "load_remote_md");
        ASSERT_MSG(std::string(agent_name) == make_agent_name(peer_id), "Loaded unexpected remote NIXL agent name");
        std::free(agent_name);
    }

    std::unordered_map<int, std::string> desc_payloads;
    for (const auto& [peer_id, peer_ptr] : peers_) {
        const auto& peer = *peer_ptr;
        NixlDescriptorEnvelope env;
        env.peer_dev_id = local_device_id_;
        for (const auto& desc : peer.recv_slot_descs) {
            env.descs.push_back(NixlDescriptorEnvelope::DescTriple{desc.addr, desc.len, desc.dev_id});
        }
        desc_payloads.emplace(peer_id, cerealize_(env));
    }
    auto remote_descs = exchange_phase(local_global_id_, peer_global_ids, ZMQ_NIXL_DESC_PORT_BASE, desc_payloads);
    for (auto& [peer_id, payload] : remote_descs) {
        auto& peer = peer_state(peer_id);
        NixlDescriptorEnvelope env;
        decerealize_(const_cast<char*>(payload.data()), payload.size(), env);
        peer.peer_dev_id = env.peer_dev_id;
        peer.remote_recv_slot_descs.clear();
        peer.remote_recv_slot_descs.reserve(env.descs.size());
        for (const auto& desc : env.descs) {
            peer.remote_recv_slot_descs.push_back(PeerState::SlotDesc{desc.addr, desc.len, desc.dev_id});
        }
    }

    std::unordered_map<int, std::string> ready_payloads;
    for (int peer_id : peer_global_ids) {
        ready_payloads.emplace(peer_id, "ready");
    }
    auto ready_acks = exchange_phase(local_global_id_, peer_global_ids, ZMQ_NIXL_READY_PORT_BASE, ready_payloads);
    ASSERT_MSG(ready_acks.size() == peer_global_ids.size(), "NIXL ready barrier failed");

    for (int peer_id : peer_global_ids) {
        check_nixl_status(nixl_capi_agent_make_connection(agent_, make_agent_name(peer_id).c_str(), backend_opts_), "make_connection");
    }

    /* See PeerState::nixl_warmup_dlist comment in nixl_context.h. Per-peer
     * prepared dlist held for the engine's lifetime; its per-worker rkey
     * unpack cache (ucp_ep_rkey_unpack via NIXL's UCX backend) is what makes
     * post_write's per-call create_xfer_req hit the fast path. */
    for (int peer_id : peer_global_ids) {
        auto& peer = peer_state(peer_id);
        check_nixl_status(nixl_capi_create_xfer_dlist(NIXL_CAPI_MEM_VRAM, &peer.nixl_warmup_dlist),
                          "create_xfer_dlist(remote-warmup)");
        for (const auto& desc : peer.remote_recv_slot_descs) {
            check_nixl_status(nixl_capi_xfer_dlist_add_desc(peer.nixl_warmup_dlist, desc.addr, desc.len, desc.dev_id),
                              "xfer_dlist_add_desc(remote-warmup)");
        }
        check_nixl_status(nixl_capi_prep_xfer_dlist(agent_, peer.peer_name.c_str(),
                                                     peer.nixl_warmup_dlist,
                                                     &peer.nixl_warmup_dlist_h, backend_opts_),
                          "prep_xfer_dlist(remote-warmup)");
    }
}

void NixlContext::shutdown() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (!initialized_) {
        return;
    }

    post_worker_stop_.store(true);
    post_cv_.notify_all();
    for (auto& th : post_workers_) {
        if (th.joinable()) {
            th.join();
        }
    }
    post_workers_.clear();

    // [V4] Stop the poll-worker thread.
    poll_worker_stop_.store(true);
    if (poll_worker_.joinable()) {
        poll_worker_.join();
    }

    for (auto& [_, peer_ptr] : peers_) {
        auto& peer = *peer_ptr;
        if (peer.nixl_warmup_dlist_h) nixl_capi_release_xfer_dlist_handle(agent_, peer.nixl_warmup_dlist_h);
        if (peer.nixl_warmup_dlist) nixl_capi_destroy_xfer_dlist(peer.nixl_warmup_dlist);
        if (peer.recv_reg_dlist) {
            nixl_capi_deregister_mem(agent_, peer.recv_reg_dlist, backend_opts_);
            nixl_capi_destroy_reg_dlist(peer.recv_reg_dlist);
        }
        if (peer.recv_buf) cudaFree(peer.recv_buf);
    }
    peers_.clear();

    if (send_warmup_dlist_h_) nixl_capi_release_xfer_dlist_handle(agent_, send_warmup_dlist_h_);
    if (send_warmup_dlist_) nixl_capi_destroy_xfer_dlist(send_warmup_dlist_);
    if (send_reg_dlist_) {
        nixl_capi_deregister_mem(agent_, send_reg_dlist_, backend_opts_);
        nixl_capi_destroy_reg_dlist(send_reg_dlist_);
    }
    if (send_buf_) cudaFree(send_buf_);
    if (recv_stream_) { cudaStreamDestroy(recv_stream_); recv_stream_ = nullptr; }
    if (cached_post_opts_) { nixl_capi_destroy_opt_args(cached_post_opts_); cached_post_opts_ = nullptr; }
    if (backend_opts_) nixl_capi_destroy_opt_args(backend_opts_);
    if (backend_) nixl_capi_destroy_backend(backend_);
    if (agent_) nixl_capi_destroy_agent(agent_);
    initialized_ = false;
}

int NixlContext::acquire_send_slot() {
    std::unique_lock<std::mutex> lock(send_mutex_);
    send_cv_.wait(lock, [&]() {
        return std::find(send_slot_busy_.begin(), send_slot_busy_.end(), false) != send_slot_busy_.end();
    });
    for (int offset = 0; offset < NIXL_SEND_RING_SIZE; ++offset) {
        int slot = (next_send_slot_ + offset) % NIXL_SEND_RING_SIZE;
        if (!send_slot_busy_[slot]) {
            send_slot_busy_[slot] = true;
            next_send_slot_ = (slot + 1) % NIXL_SEND_RING_SIZE;
            return slot;
        }
    }
    ASSERT_MSG(false, "Failed to allocate NIXL send slot");
}

void NixlContext::release_send_slot(int slot_id) {
    std::lock_guard<std::mutex> lock(send_mutex_);
    send_slot_busy_[slot_id] = false;
    send_cv_.notify_all();
}

uintptr_t NixlContext::send_slot_ptr(int slot_id) const {
    return reinterpret_cast<uintptr_t>(send_buf_) + slot_id * NIXL_MAX_BATCH_BYTES;
}

size_t NixlContext::slot_size() const {
    return NIXL_MAX_BATCH_BYTES;
}

int NixlContext::acquire_recv_slot(int peer_global_id) {
    auto& peer = peer_state(peer_global_id);
    std::unique_lock<std::mutex> lock(peer.credit_mutex);
    peer.credit_cv.wait(lock, [&]() { return !peer.available_recv_slots.empty(); });
    int slot = peer.available_recv_slots.front();
    peer.available_recv_slots.pop_front();
    peer.credits.fetch_sub(1);
    return slot;
}

void NixlContext::note_send_posted(int peer_global_id, int slot_id, int seq) {
    auto& peer = peer_state(peer_global_id);
    {
        std::lock_guard<std::mutex> lock(peer.credit_mutex);
        ASSERT_MSG(0 <= slot_id && slot_id < NIXL_RECV_RING_SIZE, "note_send_posted: slot OOR");
        auto& st = peer.slot_state[slot_id];
        ASSERT_MSG(st.expected_credit_seq == -1, "note_send_posted: remote recv slot reused before credit");
        st.expected_credit_seq = seq;
    }
    peer.sends_posted.fetch_add(1, std::memory_order_relaxed);
}

int NixlContext::next_send_seq(int peer_global_id) {
    return peer_state(peer_global_id).next_seq.fetch_add(1, std::memory_order_relaxed);
}

void NixlContext::note_credit_sent(int peer_global_id) {
    auto& peer = peer_state(peer_global_id);
    peer.credits_sent.fetch_add(1, std::memory_order_relaxed);
}

std::string NixlContext::dump_diagnostic() const {
    std::string out = "[NIXL_DIAG dev=" + std::to_string(local_global_id_) + "]";
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        int busy = 0;
        for (bool b : send_slot_busy_) if (b) ++busy;
        out += " send_slots=" + std::to_string(busy) + "/" + std::to_string(NIXL_SEND_RING_SIZE);
    }
    {
        std::lock_guard<std::mutex> lock(pending_xfers_mutex_);
        out += " pending_xfers=" + std::to_string(pending_xfers_.size());
    }
    {
        std::lock_guard<std::mutex> lock(post_mutex_);
        out += " post_queue=" + std::to_string(post_queue_.size());
    }
    {
        std::lock_guard<std::mutex> lock(notif_mutex_);
        out += " pending_arrivals=" + std::to_string(pending_arrivals_.size());
    }
    for (const auto& kv : peers_) {
        const auto& peer = *kv.second;
        size_t avail = 0;
        {
            std::lock_guard<std::mutex> lock(peer.credit_mutex);
            avail = peer.available_recv_slots.size();
        }
        const uint64_t posted = peer.sends_posted.load(std::memory_order_relaxed);
        const uint64_t credr = peer.credits_received.load(std::memory_order_relaxed);
        const uint64_t creds = peer.credits_sent.load(std::memory_order_relaxed);
        const int credits_avail = peer.credits.load(std::memory_order_relaxed);
        out += " | peer=" + std::to_string(kv.first)
            + " posted=" + std::to_string(posted)
            + " cred_recv=" + std::to_string(credr)
            + " in_flight=" + std::to_string(posted - credr)
            + " send_credits_avail=" + std::to_string(credits_avail)
            + " recv_slots_free=" + std::to_string(avail) + "/" + std::to_string(NIXL_RECV_RING_SIZE)
            + " creds_sent_back=" + std::to_string(creds);
    }
    return out;
}

void NixlContext::on_recv_credit(int peer_global_id, int slot_id, int seq) {
    auto& peer = peer_state(peer_global_id);
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(peer.credit_mutex);
        if (slot_id < 0 || slot_id >= NIXL_RECV_RING_SIZE) {
            DMOE_LOG(WARNING) << "[NIXL_C_BAD_SLOT] peer=" << peer_global_id
                << " slot=" << slot_id << " seq=" << seq << LEND;
            return;
        }
        auto& st = peer.slot_state[slot_id];
        if (st.expected_credit_seq == seq) {
            st.expected_credit_seq = -1;
            peer.available_recv_slots.push_back(slot_id);
            accepted = true;
        } else {
            DMOE_LOG(INFO) << "[NIXL_C_DUP] peer=" << peer_global_id
                << " slot=" << slot_id << " seq=" << seq
                << " expected=" << st.expected_credit_seq << LEND;
        }
    }
    if (accepted) {
        peer.credits.fetch_add(1);
        peer.credits_received.fetch_add(1, std::memory_order_relaxed);
        peer.credit_cv.notify_all();
    }
}

uintptr_t NixlContext::recv_slot_ptr(int peer_global_id, int slot_id) const {
    const auto& peer = peer_state(peer_global_id);
    return reinterpret_cast<uintptr_t>(peer.recv_buf) + slot_id * NIXL_MAX_BATCH_BYTES;
}

nixlXferReqH* NixlContext::post_write(int peer_global_id, int send_slot, int recv_slot,
                                     size_t bytes_to_write, const std::string& notif,
                                     NixlPostWriteTimings* out_timings,
                                     nixl_capi_xfer_dlist_t* out_pc_local,
                                     nixl_capi_xfer_dlist_t* out_pc_remote) {
    ASSERT_MSG(out_pc_local && out_pc_remote,
               "post_write: out_pc_local and out_pc_remote are required for dlist cleanup");
    *out_pc_local = nullptr;
    *out_pc_remote = nullptr;

    auto& peer = peer_state(peer_global_id);
    ASSERT_MSG(recv_slot >= 0 && recv_slot < (int)peer.remote_recv_slot_descs.size(),
               "post_write: recv_slot out of range for peer's remote slots");
    const auto& remote_desc = peer.remote_recv_slot_descs[recv_slot];
    ASSERT_MSG(bytes_to_write > 0, "post_write: bytes_to_write must be > 0");
    ASSERT_MSG(bytes_to_write <= remote_desc.len, "post_write: bytes_to_write exceeds remote slot size");

    const bool tracing = out_timings != nullptr;
    const double t0 = tracing ? nixl_wall_time_s() : 0.0;

    /* DESIGN_A0+B3 correctness: each post_worker thread needs its own
     * opt_args to avoid races on set_notif_msg (multiple threads writing
     * different notif payloads to the same opt_args object). The cached
     * shared cached_post_opts_ is unsafe under multi-thread posting. */
    thread_local nixl_capi_opt_args_t tl_post_opts = nullptr;
    if (tl_post_opts == nullptr) {
        check_nixl_status(nixl_capi_create_opt_args(&tl_post_opts), "create_opt_args(tl_post)");
        check_nixl_status(nixl_capi_opt_args_add_backend(tl_post_opts, backend_), "opt_args_add_backend(tl_post)");
        check_nixl_status(nixl_capi_opt_args_set_has_notif(tl_post_opts, true), "opt_args_set_has_notif(tl_post)");
    }
    check_nixl_status(nixl_capi_opt_args_set_notif_msg(tl_post_opts, notif.data(), notif.size()), "opt_args_set_notif_msg");

    const double t_pre_make = tracing ? nixl_wall_time_s() : 0.0;

    const uintptr_t local_addr = reinterpret_cast<uintptr_t>(send_buf_)
                                 + size_t(send_slot) * NIXL_MAX_BATCH_BYTES;
    const uintptr_t remote_addr = remote_desc.addr;

    nixl_capi_xfer_dlist_t pc_local = nullptr;
    nixl_capi_xfer_dlist_t pc_remote = nullptr;
    check_nixl_status(nixl_capi_create_xfer_dlist(NIXL_CAPI_MEM_VRAM, &pc_local),
                      "create_xfer_dlist(pc_local)");
    check_nixl_status(nixl_capi_xfer_dlist_add_desc(pc_local, local_addr, bytes_to_write,
                                                    static_cast<uint64_t>(local_device_id_)),
                      "xfer_dlist_add_desc(pc_local)");
    check_nixl_status(nixl_capi_create_xfer_dlist(NIXL_CAPI_MEM_VRAM, &pc_remote),
                      "create_xfer_dlist(pc_remote)");
    check_nixl_status(nixl_capi_xfer_dlist_add_desc(pc_remote, remote_addr, bytes_to_write,
                                                    remote_desc.dev_id),
                      "xfer_dlist_add_desc(pc_remote)");

    nixl_capi_xfer_req_t req = nullptr;
    check_nixl_status(nixl_capi_create_xfer_req(
        agent_, NIXL_CAPI_XFER_OP_WRITE,
        pc_local, pc_remote, peer.peer_name.c_str(),
        &req, tl_post_opts), "create_xfer_req");

    *out_pc_local = pc_local;
    *out_pc_remote = pc_remote;

    const double t_pre_post = tracing ? nixl_wall_time_s() : 0.0;
    check_nixl_status(nixl_capi_post_xfer_req(agent_, req, tl_post_opts), "post_xfer_req");
    const double t_post_done = tracing ? nixl_wall_time_s() : 0.0;

    if (tracing) {
        const double t_end = nixl_wall_time_s();
        out_timings->dt_make_s = t_pre_post - t_pre_make;
        out_timings->dt_post_s = t_post_done - t_pre_post;
        out_timings->dt_create_s = t_post_done - t_pre_make;
        out_timings->dt_other_s = (t_pre_make - t0) + (t_end - t_post_done);
    }
    return reinterpret_cast<nixl_capi_xfer_req_t>(req);
}

void NixlContext::enqueue_post(NixlPendingPost&& post) {
    {
        std::lock_guard<std::mutex> lock(post_mutex_);
        post_queue_.push_back(std::move(post));
    }
    post_cv_.notify_one();
}

void NixlContext::post_worker_loop() {
    CUDACHECK(cudaSetDevice(local_device_id_));
    while (!post_worker_stop_.load(std::memory_order_acquire)) {
        NixlPendingPost item;
        {
            std::unique_lock<std::mutex> lock(post_mutex_);
            post_cv_.wait(lock, [this]() {
                return post_worker_stop_.load(std::memory_order_acquire) || !post_queue_.empty();
            });
            if (post_worker_stop_.load(std::memory_order_acquire) && post_queue_.empty()) {
                return;
            }
            item = std::move(post_queue_.front());
            post_queue_.pop_front();
        }

        const double t_pickup = item.tracing ? nixl_wall_time_s() : 0.0;
        int spin_count = 0;
        while (true) {
            cudaError_t qerr = cudaEventQuery(item.d2d_event);
            if (qerr == cudaSuccess) break;
            if (qerr != cudaErrorNotReady) {
                DMOE_LOG(ERROR) << "post_worker: cudaEventQuery failed: "
                                << cudaGetErrorString(qerr) << LEND;
                ASSERT_MSG(false, "post_worker: cudaEventQuery failed");
            }
            if (++spin_count > 64) {
                std::this_thread::yield();
                spin_count = 0;
            }
        }
        cudaEventDestroy(item.d2d_event);

        NixlPostWriteTimings timings;
        nixl_capi_xfer_dlist_t pc_local = nullptr;
        nixl_capi_xfer_dlist_t pc_remote = nullptr;
        const double t_pre_post = item.tracing ? nixl_wall_time_s() : 0.0;
        nixlXferReqH* handle = post_write(
            item.peer_id, item.send_slot, item.recv_slot,
            item.bytes_to_write, item.notif_payload,
            item.tracing ? &timings : nullptr,
            &pc_local, &pc_remote);
        const double t_post_done = item.tracing ? nixl_wall_time_s() : 0.0;

        enqueue_pending_xfer(handle, item.send_slot, item.notif_payload,
                             item.peer_id, item.recv_slot, item.seq,
                             pc_local, pc_remote);

        if (item.tracing && tracing_enabled()) {
            record_send_trace(NixlSendTraceTuple{
                item.peer_id,
                item.seq,
                item.recv_slot,
                item.bytes_to_write,
                item.t_enter_send_s,
                item.dt_acq_recv_s,
                item.dt_acq_send_s,
                item.dt_d2d_enqueue_s,
                item.dt_build_notif_s,
                item.dt_dispatcher_total_s,
                t_pickup - item.t_enqueued_s,
                t_pre_post - t_pickup,
                timings.dt_create_s,
                timings.dt_make_s,
                timings.dt_post_s,
                timings.dt_other_s,
                t_post_done - item.t_enter_send_s,
            });
        }
    }
}

int NixlContext::xfer_status(nixlXferReqH* h) {
    auto status = nixl_capi_get_xfer_status(agent_, reinterpret_cast<nixl_capi_xfer_req_t>(h));
    if (status == NIXL_CAPI_SUCCESS) return 0;
    if (status == NIXL_CAPI_IN_PROG) return 1;
    return static_cast<int>(status);
}

void NixlContext::release_xfer(nixlXferReqH* h) {
    check_nixl_status(nixl_capi_release_xfer_req(agent_, reinterpret_cast<nixl_capi_xfer_req_t>(h)), "release_xfer_req");
    check_nixl_status(nixl_capi_destroy_xfer_req(reinterpret_cast<nixl_capi_xfer_req_t>(h)), "destroy_xfer_req");
}

void NixlContext::enqueue_pending_xfer(nixlXferReqH* handle, int send_slot, std::string notif_keepalive,
                                       int peer_id, int recv_slot, int seq,
                                       nixl_capi_xfer_dlist_t pc_local,
                                       nixl_capi_xfer_dlist_t pc_remote) {
    std::lock_guard<std::mutex> lock(pending_xfers_mutex_);
    pending_xfers_.push_back(PendingXfer{handle, send_slot, std::move(notif_keepalive),
                                         peer_id, recv_slot, seq, pc_local, pc_remote});
}

bool NixlContext::has_pending_xfers() const {
    std::lock_guard<std::mutex> lock(pending_xfers_mutex_);
    return !pending_xfers_.empty();
}

int NixlContext::drain_pending_xfers() {
    std::lock_guard<std::mutex> lock(pending_xfers_mutex_);
    int released = 0;
    auto it = pending_xfers_.begin();
    while (it != pending_xfers_.end()) {
        const int status = xfer_status(it->handle);
        if (status == 0) {
            release_xfer(it->handle);
            if (it->pc_local) nixl_capi_destroy_xfer_dlist(it->pc_local);
            if (it->pc_remote) nixl_capi_destroy_xfer_dlist(it->pc_remote);
            release_send_slot(it->send_slot);
            it = pending_xfers_.erase(it);
            ++released;
        } else if (status == 1) {
            ++it;
        } else {
            DMOE_LOG(ERROR) << "NIXL xfer failed in drain_pending_xfers, status=" << status
                            << " send_slot=" << it->send_slot
                            << " peer=" << it->peer_id
                            << " recv_slot=" << it->recv_slot
                            << " seq=" << it->seq
                            << " notif_len=" << it->notif_keepalive.size() << LEND;
            release_xfer(it->handle);
            if (it->pc_local) nixl_capi_destroy_xfer_dlist(it->pc_local);
            if (it->pc_remote) nixl_capi_destroy_xfer_dlist(it->pc_remote);
            release_send_slot(it->send_slot);
            it = pending_xfers_.erase(it);
            ++released;
        }
    }
    return released;
}

// [V2-A][V2-D] Binary notif parser. 'D' (data) notifs carry a header plus
// cerealized BatchMetadata and become NixlMetaArrival entries. 'C' (credit)
// notifs are header-only and immediately release a recv slot.
void NixlContext::poll_notifs() {
    nixl_capi_notif_map_t notif_map = nullptr;
    check_nixl_status(nixl_capi_create_notif_map(&notif_map), "create_notif_map");
    check_nixl_status(nixl_capi_get_notifs(agent_, notif_map, backend_opts_), "get_notifs");

    size_t agent_count = 0;
    check_nixl_status(nixl_capi_notif_map_size(notif_map, &agent_count), "notif_map_size");

    std::vector<NixlMetaArrival> new_arrivals;
    std::vector<std::tuple<int, int, int>> credit_returns;

    for (size_t i = 0; i < agent_count; ++i) {
        const char* sender_name = nullptr;
        check_nixl_status(nixl_capi_notif_map_get_agent_at(notif_map, i, &sender_name), "notif_map_get_agent_at");
        int sender_id = parse_agent_rank(sender_name ? sender_name : "");
        if (sender_id < 0) {
            continue;
        }
        size_t notif_count = 0;
        check_nixl_status(nixl_capi_notif_map_get_notifs_size(notif_map, sender_name, &notif_count), "notif_map_get_notifs_size");
        for (size_t j = 0; j < notif_count; ++j) {
            const void* data = nullptr;
            size_t len = 0;
            check_nixl_status(nixl_capi_notif_map_get_notif(notif_map, sender_name, j, &data, &len), "notif_map_get_notif");
            if (data == nullptr || len < sizeof(NixlNotifHeader)) {
                continue;
            }
            NixlNotifHeader hdr;
            std::memcpy(&hdr, data, sizeof(hdr));
            if (hdr.type == NIXL_NOTIF_CREDIT) {
                credit_returns.emplace_back(sender_id, static_cast<int>(hdr.slot_id), static_cast<int>(hdr.seq));
            } else if (hdr.type == NIXL_NOTIF_DATA) {
                if (len < sizeof(NixlNotifHeader) + hdr.meta_len) {
                    DMOE_LOG(ERROR) << "Truncated NIXL D notif: len=" << len
                                    << " meta_len=" << hdr.meta_len << LEND;
                    continue;
                }
                if (!accept_data_notif(sender_id, static_cast<int>(hdr.slot_id), static_cast<int>(hdr.seq))) {
                    continue;
                }
                NixlMetaArrival arrival;
                arrival.peer_id = sender_id;
                arrival.seq = static_cast<int>(hdr.seq);
                arrival.slot_id = static_cast<int>(hdr.slot_id);
                arrival.t_arrived_s = nixl_wall_time_s();
                if (hdr.meta_len > 0) {
                    auto meta_ptr = std::make_shared<BatchMetadata>();
                    char* payload = const_cast<char*>(reinterpret_cast<const char*>(data)) + sizeof(NixlNotifHeader);
                    decerealize_(payload, hdr.meta_len, *meta_ptr);
                    arrival.meta = std::move(meta_ptr);
                }
                new_arrivals.push_back(std::move(arrival));
            }
        }
    }
    nixl_capi_destroy_notif_map(notif_map);

    if (!new_arrivals.empty()) {
        std::lock_guard<std::mutex> lock(notif_mutex_);
        const int added = static_cast<int>(new_arrivals.size());
        for (auto& arrival : new_arrivals) {
            pending_arrivals_.push_back(std::move(arrival));
        }
        arrival_count_.fetch_add(added, std::memory_order_release);
    }
    for (auto& [peer, slot, seq] : credit_returns) {
        on_recv_credit(peer, slot, seq);
    }
}

// [V2-A]
std::vector<NixlMetaArrival> NixlContext::drain_meta_arrivals() {
    std::lock_guard<std::mutex> lock(notif_mutex_);
    if (pending_arrivals_.empty()) {
        return {};
    }
    std::vector<NixlMetaArrival> out;
    out.swap(pending_arrivals_);
    arrival_count_.store(0, std::memory_order_release);
    return out;
}

// [V3] Credit notif carries (slot_id, seq). seq must match the sender's
// expected_credit_seq for that slot before the slot is freed. NIXL notifs
// are reliably delivered on healthy IB/RoCE/TCP, but reordered and
// arbitrarily delayed; without seq matching, a late C from an earlier
// reuse of the same slot could falsely ack a later send.
void NixlContext::send_credit(int peer_global_id, int slot_id, int seq) {
    NixlNotifHeader hdr{};
    hdr.type = NIXL_NOTIF_CREDIT;
    hdr.seq = static_cast<uint32_t>(seq);
    hdr.slot_id = static_cast<uint32_t>(slot_id);
    hdr.meta_len = 0;
    check_nixl_status(nixl_capi_gen_notif(
        agent_, make_agent_name(peer_global_id).c_str(),
        reinterpret_cast<const char*>(&hdr), sizeof(hdr), backend_opts_), "gen_notif(credit)");
    note_credit_sent(peer_global_id);
}

bool NixlContext::accept_data_notif(int peer_global_id, int slot_id, int seq) {
    auto& peer = peer_state(peer_global_id);
    std::lock_guard<std::mutex> lock(peer.credit_mutex);
    if (slot_id < 0 || slot_id >= NIXL_RECV_RING_SIZE) {
        DMOE_LOG(WARNING) << "[NIXL_D_BAD_SLOT] peer=" << peer_global_id
            << " slot=" << slot_id << " seq=" << seq << LEND;
        return false;
    }
    auto& st = peer.slot_state[slot_id];
    if (st.inflight_recv_seq == seq || st.completed_recv_seq >= seq) {
        DMOE_LOG(INFO) << "[NIXL_D_DUP] peer=" << peer_global_id
            << " slot=" << slot_id << " seq=" << seq
            << " inflight=" << st.inflight_recv_seq
            << " completed=" << st.completed_recv_seq << LEND;
        return false;
    }
    if (st.inflight_recv_seq != -1) {
        DMOE_LOG(WARNING) << "[NIXL_D_REUSE] peer=" << peer_global_id
            << " slot=" << slot_id << " seq=" << seq
            << " prior_inflight=" << st.inflight_recv_seq << LEND;
    }
    st.inflight_recv_seq = seq;
    return true;
}

void NixlContext::mark_recv_completed(int peer_global_id, int slot_id, int seq) {
    if (slot_id < 0) return;
    auto& peer = peer_state(peer_global_id);
    std::lock_guard<std::mutex> lock(peer.credit_mutex);
    if (slot_id >= NIXL_RECV_RING_SIZE) return;
    auto& st = peer.slot_state[slot_id];
    if (st.completed_recv_seq < seq) st.completed_recv_seq = seq;
    if (st.inflight_recv_seq == seq) st.inflight_recv_seq = -1;
}

// [V4] Dedicated notif polling thread. RDMA + UCX deliver notifs reliably;
// they sit in a user-space queue at the receiver until get_notifs() drains
// them. This thread does that draining so the queue stays empty at sub-ms
// latency, regardless of what the pool/dispatcher threads are doing.
//
// Backoff: when a poll yields nothing, sleep ~POLL_IDLE_SLEEP_US to cap the
// poll rate. Without this cap, a tight loop of get_notifs() at multi-MHz
// triggers a known NIXL library bug (double-free in nixlAgent::getNotifs
// observed at ~2.5M calls/sec, see HANDOFF_MAY3.md). 10us yields ~100K
// polls/sec, well below the trigger and 20x faster than the previous
// in-loop throttle's best case (~28K/sec).
void NixlContext::poll_worker_loop() {
    constexpr int POLL_IDLE_SLEEP_US = 10;
    while (!poll_worker_stop_.load(std::memory_order_acquire)) {
        size_t total_credits_before = 0;
        for (auto& kv : peers_) {
            total_credits_before += kv.second->credits_received.load(std::memory_order_relaxed);
        }
        const int prev_arrivals = arrival_count_.load(std::memory_order_relaxed);

        poll_notifs();

        const int new_arrivals = arrival_count_.load(std::memory_order_relaxed);
        size_t total_credits_after = 0;
        for (auto& kv : peers_) {
            total_credits_after += kv.second->credits_received.load(std::memory_order_relaxed);
        }
        const bool had_work = (new_arrivals != prev_arrivals)
                           || (total_credits_after != total_credits_before);
        if (!had_work) {
            std::this_thread::sleep_for(std::chrono::microseconds(POLL_IDLE_SLEEP_US));
        }
    }
}

void NixlContext::record_send_trace(const NixlSendTraceTuple& trace) {
    std::lock_guard<std::mutex> lock(send_traces_mutex_);
    send_traces_.push_back(trace);
}

void NixlContext::record_recv_trace(const NixlRecvTraceTuple& trace) {
    std::lock_guard<std::mutex> lock(recv_traces_mutex_);
    recv_traces_.push_back(trace);
}

std::vector<NixlSendTraceTuple> NixlContext::drain_send_traces() {
    std::lock_guard<std::mutex> lock(send_traces_mutex_);
    auto out = std::move(send_traces_);
    send_traces_.clear();
    return out;
}

std::vector<NixlRecvTraceTuple> NixlContext::drain_recv_traces() {
    std::lock_guard<std::mutex> lock(recv_traces_mutex_);
    auto out = std::move(recv_traces_);
    recv_traces_.clear();
    return out;
}

#endif
