/*
 * shared_agent_mesh.cu
 *
 * Engine-architecture reproducer: ONE NIXL agent per rank, with N connections
 * to N remote ranks. All N EPs share the same ucp_workers and the same agent
 * lock — exactly as the real engine does it.
 *
 * Runs against 6 remote nodes (sgpu3, sgpu4, sgpu5, sgpu7, sgpu8, sgpu9).
 *
 * Modes:
 *   --rank 0 --num-peers N --inflight K [--bidirectional]
 *       active poster: round-robin posts across N connections, keeps K
 *       outstanding requests, measures post latency.
 *   --rank 1 --peer-idx P
 *       passive responder for slot P. Optionally also actively sends back
 *       (bidirectional) at rate determined by --rank1-inflight.
 *
 * Fix modes (controlled via --fix-mode, all use the same registered buffers):
 *
 *   current      [baseline, default]
 *       Use prepped dlist handles with descriptors sized to --desc-len-bytes.
 *       Transfers FULL descriptor length per post (the bug). Reproduces the
 *       4 MiB-at-engine 5 ms latency when --desc-len-bytes=4194304.
 *
 *   tight-fixed
 *       Same as current, but documents that --desc-len-bytes is intentionally
 *       small (e.g. 65536). Still transfers full descriptor length, but with
 *       a smaller fixed length. Wastes bandwidth when payload < desc_len, and
 *       hard-caps max payload.
 *
 *   per-call
 *       Build a fresh 1-elem RAW dlist for both local and remote sides every
 *       post, sized to --bytes-to-write. Use nixl_capi_create_xfer_req
 *       (NOT make_xfer_req) which takes raw dlists + remote_agent name and
 *       does not require prep. Destroy the dlists after the request completes.
 *
 *   log2-pool
 *       At init, pre-create K dlists at log2-spaced sizes (default
 *       4K/16K/64K/256K/1M/4M, tunable via --pool-sizes). Each pool dlist has
 *       send_ring/recv_ring descriptors at the same starting addresses, but
 *       sized to that pool's bytes. Per-post, pick smallest pool size >=
 *       --bytes-to-write and use its prepped handle. Constant per-post
 *       overhead (one branch + one indexing op) and zero hot-path allocation.
 *
 * --bytes-to-write B           single payload size (default = desc_len)
 * --bytes-to-write-list B1,B2  round-robin payload sizes (overrides single)
 * --pool-sizes 4096,16384,...  sizes for log2-pool mode
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "wrapper.h"

#define CHECK_CUDA(call) do {                                                 \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
        std::cerr << "FATAL cuda: " << #call << " -> "                        \
                  << cudaGetErrorString(_e) << "\n";                          \
        std::exit(5);                                                         \
    }                                                                         \
} while (0)

#define CHECK_NIXL(call, label) do {                                          \
    nixl_capi_status_t _s = (call);                                           \
    if (_s != NIXL_CAPI_SUCCESS) {                                            \
        std::cerr << "FATAL nixl: " << label << " status=" << _s << "\n";     \
        std::exit(3);                                                         \
    }                                                                         \
} while (0)

namespace {

constexpr size_t NIXL_MAX_BATCH_BYTES = 4ull * 1024 * 1024;
constexpr int RV_PORT_BASE = 48000;

enum class FixMode { CURRENT, TIGHT_FIXED, PER_CALL, LOG2_POOL };

const char* fix_mode_str(FixMode m) {
    switch (m) {
        case FixMode::CURRENT:     return "current";
        case FixMode::TIGHT_FIXED: return "tight-fixed";
        case FixMode::PER_CALL:    return "per-call";
        case FixMode::LOG2_POOL:   return "log2-pool";
    }
    return "?";
}

FixMode parse_fix_mode(const std::string& s) {
    if (s == "current")     return FixMode::CURRENT;
    if (s == "tight-fixed") return FixMode::TIGHT_FIXED;
    if (s == "per-call")    return FixMode::PER_CALL;
    if (s == "log2-pool")   return FixMode::LOG2_POOL;
    std::cerr << "FATAL: bad --fix-mode " << s << "\n";
    std::exit(2);
}

struct Args {
    int rank = -1;
    std::string peer_host;
    int port_offset = 0;
    int num_peers = 1;
    int inflight = 1;
    int rank1_inflight = 0;
    int iters = 5000;
    int warmup = 500;
    std::string mem = "vram";
    int cuda_device = 0;
    std::string out_csv;
    int peer_idx = -1;
    std::string peer_host_list;
    int num_workers = 4;
    int send_ring_size = 32;
    int recv_ring_size = 8;
    size_t desc_len = NIXL_MAX_BATCH_BYTES;

    // [FIX MODE EXTENSIONS]
    FixMode fix_mode = FixMode::CURRENT;
    std::vector<size_t> bytes_list;     // populated by --bytes-to-write or --bytes-to-write-list
    std::vector<size_t> pool_sizes;     // for log2-pool, populated by --pool-sizes
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "FATAL: " << msg << " (errno=" << errno << ": "
              << std::strerror(errno) << ")\n";
    std::exit(2);
}

static inline uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1'000'000'000ull + uint64_t(ts.tv_nsec);
}

ssize_t send_all(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t left = len;
    while (left > 0) {
        ssize_t n = ::send(fd, p, left, 0);
        if (n <= 0) { if (errno == EINTR) continue; return -1; }
        p += n; left -= n;
    }
    return ssize_t(len);
}

ssize_t recv_all(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t left = len;
    while (left > 0) {
        ssize_t n = ::recv(fd, p, left, 0);
        if (n <= 0) { if (n < 0 && errno == EINTR) continue; return -1; }
        p += n; left -= n;
    }
    return ssize_t(len);
}

std::string exchange_blob_tcp(const std::string& peer_host, int port,
                              int rank, const std::string& my_blob) {
    if (rank == 0) {
        int srv = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) die("socket");
        int reuse = 1;
        ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(port);
        a.sin_addr.s_addr = INADDR_ANY;
        if (::bind(srv, (sockaddr*)&a, sizeof(a)) < 0) die("bind");
        if (::listen(srv, 1) < 0) die("listen");
        int c = ::accept(srv, nullptr, nullptr);
        if (c < 0) die("accept");
        ::close(srv);
        uint32_t mlen = htonl((uint32_t)my_blob.size());
        send_all(c, &mlen, 4);
        send_all(c, my_blob.data(), my_blob.size());
        uint32_t rlen_n = 0;
        recv_all(c, &rlen_n, 4);
        uint32_t rlen = ntohl(rlen_n);
        std::string remote(rlen, '\0');
        recv_all(c, remote.data(), rlen);
        ::close(c);
        return remote;
    } else {
        int c = -1;
        for (int attempt = 0; attempt < 600; ++attempt) {
            c = ::socket(AF_INET, SOCK_STREAM, 0);
            if (c < 0) die("socket");
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons(port);
            ::inet_pton(AF_INET, peer_host.c_str(), &a.sin_addr);
            if (::connect(c, (sockaddr*)&a, sizeof(a)) == 0) break;
            ::close(c); c = -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (c < 0) die("connect");
        uint32_t rlen_n = 0;
        recv_all(c, &rlen_n, 4);
        uint32_t rlen = ntohl(rlen_n);
        std::string remote(rlen, '\0');
        recv_all(c, remote.data(), rlen);
        uint32_t mlen = htonl((uint32_t)my_blob.size());
        send_all(c, &mlen, 4);
        send_all(c, my_blob.data(), my_blob.size());
        ::close(c);
        return remote;
    }
}

double pct(std::vector<uint64_t>& xs, double q) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    if (xs.size() == 1) return double(xs[0]);
    double pos = (xs.size() - 1) * q;
    size_t lo = size_t(pos);
    size_t hi = std::min(lo + 1, xs.size() - 1);
    double frac = pos - double(lo);
    return double(xs[lo]) * (1.0 - frac) + double(xs[hi]) * frac;
}

std::vector<size_t> parse_size_list(const std::string& s) {
    std::vector<size_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back((size_t)std::stoll(item));
    }
    return out;
}

void parse_args(int argc, char** argv, Args& a) {
    std::string single_bytes;
    std::string bytes_list_str;
    std::string pool_sizes_str;
    std::string fix_mode_str_in;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* tag) -> std::string {
            if (i + 1 >= argc) die(std::string(tag) + " requires a value");
            return argv[++i];
        };
        if (s == "--rank") a.rank = std::stoi(next("--rank"));
        else if (s == "--peer-host") a.peer_host = next("--peer-host");
        else if (s == "--port-offset") a.port_offset = std::stoi(next("--port-offset"));
        else if (s == "--num-peers") a.num_peers = std::stoi(next("--num-peers"));
        else if (s == "--inflight") a.inflight = std::stoi(next("--inflight"));
        else if (s == "--rank1-inflight") a.rank1_inflight = std::stoi(next("--rank1-inflight"));
        else if (s == "--iters") a.iters = std::stoi(next("--iters"));
        else if (s == "--warmup") a.warmup = std::stoi(next("--warmup"));
        else if (s == "--mem") a.mem = next("--mem");
        else if (s == "--cuda-device") a.cuda_device = std::stoi(next("--cuda-device"));
        else if (s == "--out-csv") a.out_csv = next("--out-csv");
        else if (s == "--peer-idx") a.peer_idx = std::stoi(next("--peer-idx"));
        else if (s == "--peer-host-list") a.peer_host_list = next("--peer-host-list");
        else if (s == "--num-workers") a.num_workers = std::stoi(next("--num-workers"));
        else if (s == "--send-ring-size") a.send_ring_size = std::stoi(next("--send-ring-size"));
        else if (s == "--recv-ring-size") a.recv_ring_size = std::stoi(next("--recv-ring-size"));
        else if (s == "--desc-len-mib") a.desc_len = (size_t)std::stoll(next("--desc-len-mib")) * 1024 * 1024;
        else if (s == "--desc-len-bytes") a.desc_len = (size_t)std::stoll(next("--desc-len-bytes"));
        else if (s == "--fix-mode") fix_mode_str_in = next("--fix-mode");
        else if (s == "--bytes-to-write") single_bytes = next("--bytes-to-write");
        else if (s == "--bytes-to-write-list") bytes_list_str = next("--bytes-to-write-list");
        else if (s == "--pool-sizes") pool_sizes_str = next("--pool-sizes");
        else die("unknown arg: " + s);
    }
    if (a.rank < 0) die("--rank required (0 or 1)");
    if (a.peer_host.empty()) die("--peer-host required");

    if (!fix_mode_str_in.empty()) a.fix_mode = parse_fix_mode(fix_mode_str_in);

    // Resolve bytes_list. Priority: --bytes-to-write-list > --bytes-to-write > desc_len.
    if (!bytes_list_str.empty()) {
        a.bytes_list = parse_size_list(bytes_list_str);
    } else if (!single_bytes.empty()) {
        a.bytes_list.push_back((size_t)std::stoll(single_bytes));
    } else {
        a.bytes_list.push_back(a.desc_len);
    }
    for (size_t b : a.bytes_list) {
        if (b == 0) die("--bytes-to-write must be > 0");
        if (b > a.desc_len) die("--bytes-to-write must be <= --desc-len-bytes");
    }

    // Resolve pool_sizes for log2-pool mode.
    if (!pool_sizes_str.empty()) {
        a.pool_sizes = parse_size_list(pool_sizes_str);
    } else {
        // Default log2-spaced pool that covers engine's 8-40 KiB range with
        // finer granularity at the small end.
        a.pool_sizes = {8 * 1024, 16 * 1024, 64 * 1024, 256 * 1024,
                        1 * 1024 * 1024, 4 * 1024 * 1024};
    }
    std::sort(a.pool_sizes.begin(), a.pool_sizes.end());
    for (size_t s : a.pool_sizes) {
        if (s == 0) die("--pool-sizes entries must be > 0");
    }
    if (a.fix_mode == FixMode::LOG2_POOL) {
        std::vector<size_t> filtered;
        for (size_t s : a.pool_sizes) if (s <= a.desc_len) filtered.push_back(s);
        if (filtered.empty()) die("no --pool-sizes entries fit within --desc-len-bytes");
        a.pool_sizes = filtered;
        for (size_t b : a.bytes_list) {
            if (b > a.pool_sizes.back())
                die("bytes-to-write exceeds largest pool size for log2-pool mode");
        }
    }
}

// Picks smallest pool index k such that pool_sizes[k] >= bytes. Returns
// pool_sizes.size()-1 if none fit (caller should have validated).
inline int pick_pool(const std::vector<size_t>& pool_sizes, size_t bytes) {
    for (size_t k = 0; k < pool_sizes.size(); ++k) {
        if (pool_sizes[k] >= bytes) return (int)k;
    }
    return (int)pool_sizes.size() - 1;
}

struct Connection {
    std::string remote_name;
    nixl_capi_xfer_dlist_handle_t remote_dlist_h = nullptr;
    // For log2-pool mode: one prepped handle per pool size.
    std::vector<nixl_capi_xfer_dlist_handle_t> remote_pool_h;
    // For per-call mode: the remote registered buffer base address, used to
    // construct fresh raw dlists per post.
    uintptr_t remote_buf_addr = 0;
};

}  // namespace

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);

    bool use_vram = (args.mem == "vram");
    if (args.mem != "dram" && args.mem != "vram")
        die("--mem must be dram or vram");

    int my_peer_idx = (args.rank == 1 && args.peer_idx >= 0) ? args.peer_idx : 0;
    int rendez_port_base = RV_PORT_BASE + args.port_offset;

    std::string my_name = std::string("shamesh-rank") + std::to_string(args.rank);
    if (args.rank == 1) my_name += "-slot" + std::to_string(my_peer_idx);

    nixl_capi_agent_config_t cfg{};
    cfg.enable_prog_thread = true;
    cfg.enable_listen_thread = true;
    cfg.listen_port = (rendez_port_base % 1000) * 30 + 49000
                      + 4 * my_peer_idx + 2 * args.rank;
    cfg.lthr_delay_us = 100000;
    cfg.pthr_delay_us = 0;
    cfg.thread_sync = NIXL_CAPI_THREAD_SYNC_RW;
    cfg.num_workers = args.num_workers;

    nixl_capi_agent_t agent = nullptr;
    CHECK_NIXL(nixl_capi_create_configured_agent(my_name.c_str(), &cfg, &agent),
               "create_configured_agent");

    nixl_capi_mem_list_t mems = nullptr;
    nixl_capi_params_t params = nullptr;
    CHECK_NIXL(nixl_capi_get_plugin_params(agent, "UCX", &mems, &params),
               "get_plugin_params");
    {
        char buf[16]; std::snprintf(buf, sizeof(buf), "%d", args.num_workers);
        CHECK_NIXL(nixl_capi_params_add(params, "num_workers", buf),
                   "params_add(num_workers)");
    }
    nixl_capi_backend_t backend = nullptr;
    CHECK_NIXL(nixl_capi_create_backend(agent, "UCX", params, &backend),
               "create_backend");
    nixl_capi_destroy_mem_list(mems);
    nixl_capi_destroy_params(params);

    nixl_capi_opt_args_t backend_opts = nullptr;
    CHECK_NIXL(nixl_capi_create_opt_args(&backend_opts), "create_opt_args");
    CHECK_NIXL(nixl_capi_opt_args_add_backend(backend_opts, backend),
               "opt_args_add_backend");
    CHECK_NIXL(nixl_capi_opt_args_set_has_notif(backend_opts, false),
               "set_has_notif(false)");

    nixl_capi_mem_type_t mem_type = use_vram ? NIXL_CAPI_MEM_VRAM
                                             : NIXL_CAPI_MEM_DRAM;
    uint64_t mem_dev_id = use_vram ? uint64_t(args.cuda_device) : 0;

    int local_ring = (args.rank == 0) ? args.send_ring_size : args.recv_ring_size;
    const size_t local_buf_size = size_t(local_ring) * args.desc_len;
    std::cerr << "rank " << args.rank << ": fix_mode=" << fix_mode_str(args.fix_mode)
              << " desc_len=" << args.desc_len
              << " local_buf=" << local_buf_size / (1024 * 1024) << " MiB ("
              << local_ring << " slots x " << args.desc_len / 1024 << " KiB each)\n";

    void* buf = nullptr;
    if (use_vram) {
        CHECK_CUDA(cudaSetDevice(args.cuda_device));
        CHECK_CUDA(cudaMalloc(&buf, local_buf_size));
        CHECK_CUDA(cudaMemset(buf, 0xa5, local_buf_size));
    } else {
        if (::posix_memalign(&buf, 4096, local_buf_size) != 0) die("posix_memalign");
        std::memset(buf, 0xa5, local_buf_size);
    }

    nixl_capi_reg_dlist_t reg_dlist = nullptr;
    CHECK_NIXL(nixl_capi_create_reg_dlist(mem_type, &reg_dlist),
               "create_reg_dlist");
    CHECK_NIXL(nixl_capi_reg_dlist_add_desc(reg_dlist, (uintptr_t)buf,
                                            local_buf_size, mem_dev_id, nullptr, 0),
               "reg_dlist_add_desc");
    CHECK_NIXL(nixl_capi_register_mem(agent, reg_dlist, backend_opts),
               "register_mem");

    // === Build local dlist(s) ===
    // For current/tight-fixed/per-call: one prepped dlist with desc_len descriptors.
    // (per-call uses raw dlists per post; the prepped one is unused but cheap to keep.)
    // For log2-pool: K prepped dlists, each with local_ring descriptors at the same
    // starting addresses but with size pool_sizes[k].
    nixl_capi_xfer_dlist_t local_dlist = nullptr;
    CHECK_NIXL(nixl_capi_create_xfer_dlist(mem_type, &local_dlist),
               "create_xfer_dlist(local)");
    for (int i = 0; i < local_ring; ++i) {
        uintptr_t addr = (uintptr_t)buf + size_t(i) * args.desc_len;
        CHECK_NIXL(nixl_capi_xfer_dlist_add_desc(local_dlist, addr, args.desc_len, mem_dev_id),
                   "xfer_dlist_add_desc(local)");
    }
    nixl_capi_xfer_dlist_handle_t local_dlist_h = nullptr;
    CHECK_NIXL(nixl_capi_prep_xfer_dlist(agent, "", local_dlist, &local_dlist_h,
                                         backend_opts),
               "prep_xfer_dlist(local)");

    // log2-pool extra local dlists
    std::vector<nixl_capi_xfer_dlist_t> local_pool_dlist;
    std::vector<nixl_capi_xfer_dlist_handle_t> local_pool_h;
    if (args.fix_mode == FixMode::LOG2_POOL) {
        local_pool_dlist.resize(args.pool_sizes.size());
        local_pool_h.resize(args.pool_sizes.size());
        for (size_t k = 0; k < args.pool_sizes.size(); ++k) {
            CHECK_NIXL(nixl_capi_create_xfer_dlist(mem_type, &local_pool_dlist[k]),
                       "create_xfer_dlist(local-pool)");
            for (int i = 0; i < local_ring; ++i) {
                uintptr_t addr = (uintptr_t)buf + size_t(i) * args.desc_len;
                CHECK_NIXL(nixl_capi_xfer_dlist_add_desc(local_pool_dlist[k], addr,
                                                          args.pool_sizes[k], mem_dev_id),
                           "xfer_dlist_add_desc(local-pool)");
            }
            CHECK_NIXL(nixl_capi_prep_xfer_dlist(agent, "",
                                                  local_pool_dlist[k],
                                                  &local_pool_h[k],
                                                  backend_opts),
                       "prep_xfer_dlist(local-pool)");
            std::cerr << "rank " << args.rank << ": log2-pool[k=" << k << "] size="
                      << args.pool_sizes[k] / 1024 << " KiB prepped\n";
        }
    }

    void* my_md = nullptr;
    size_t my_md_len = 0;
    CHECK_NIXL(nixl_capi_get_local_md(agent, &my_md, &my_md_len),
               "get_local_md");
    std::string my_md_blob((const char*)my_md, my_md_len);
    uintptr_t my_buf_addr = (uintptr_t)buf;

    std::vector<std::string> per_peer_host(args.num_peers, args.peer_host);
    if (args.rank == 0 && !args.peer_host_list.empty()) {
        per_peer_host.clear();
        std::stringstream ss(args.peer_host_list);
        std::string item;
        while (std::getline(ss, item, ',')) per_peer_host.push_back(item);
        if ((int)per_peer_host.size() != args.num_peers)
            die("--peer-host-list must have num_peers entries");
    }

    int n_conn = (args.rank == 0) ? args.num_peers : 1;
    std::vector<Connection> conns(n_conn);

    for (int slot = 0; slot < n_conn; ++slot) {
        int conn_peer_idx = (args.rank == 0) ? slot : my_peer_idx;
        int md_port = rendez_port_base + 100 * conn_peer_idx;
        std::string this_peer_host = (args.rank == 0)
                                     ? per_peer_host[slot]
                                     : args.peer_host;
        std::cerr << "rank " << args.rank << " conn-slot " << slot
                  << " peer_idx=" << conn_peer_idx
                  << ": exchange md on " << this_peer_host << ":" << md_port
                  << "\n";

        std::string remote_md = exchange_blob_tcp(this_peer_host, md_port,
                                                  args.rank, my_md_blob);
        char* remote_name_c = nullptr;
        CHECK_NIXL(nixl_capi_load_remote_md(agent, remote_md.data(),
                                            remote_md.size(), &remote_name_c),
                   "load_remote_md");
        conns[slot].remote_name = remote_name_c ? remote_name_c : "";

        char addr_blob[sizeof(uintptr_t)];
        std::memcpy(addr_blob, &my_buf_addr, sizeof(uintptr_t));
        std::string my_addr_str(addr_blob, sizeof(uintptr_t));
        std::string remote_addr_str = exchange_blob_tcp(this_peer_host,
                                                        md_port + 1,
                                                        args.rank, my_addr_str);
        uintptr_t remote_buf_addr = 0;
        std::memcpy(&remote_buf_addr, remote_addr_str.data(), sizeof(uintptr_t));
        conns[slot].remote_buf_addr = remote_buf_addr;

        CHECK_NIXL(nixl_capi_agent_make_connection(agent,
                                                   conns[slot].remote_name.c_str(),
                                                   backend_opts),
                   "make_connection");

        int remote_ring = (args.rank == 0) ? args.recv_ring_size : args.send_ring_size;

        // Per-conn baseline remote dlist (all modes prep this so make_xfer_req
        // works for current/tight-fixed; per-call ignores it).
        nixl_capi_xfer_dlist_t remote_dlist = nullptr;
        CHECK_NIXL(nixl_capi_create_xfer_dlist(mem_type, &remote_dlist),
                   "create_xfer_dlist(remote)");
        for (int i = 0; i < remote_ring; ++i) {
            uintptr_t addr = remote_buf_addr + size_t(i) * args.desc_len;
            CHECK_NIXL(nixl_capi_xfer_dlist_add_desc(remote_dlist, addr, args.desc_len, mem_dev_id),
                       "xfer_dlist_add_desc(remote)");
        }
        CHECK_NIXL(nixl_capi_prep_xfer_dlist(agent,
                                             conns[slot].remote_name.c_str(),
                                             remote_dlist,
                                             &conns[slot].remote_dlist_h,
                                             backend_opts),
                   "prep_xfer_dlist(remote)");

        // Per-conn log2-pool remote dlists.
        if (args.fix_mode == FixMode::LOG2_POOL) {
            conns[slot].remote_pool_h.resize(args.pool_sizes.size());
            for (size_t k = 0; k < args.pool_sizes.size(); ++k) {
                nixl_capi_xfer_dlist_t rd = nullptr;
                CHECK_NIXL(nixl_capi_create_xfer_dlist(mem_type, &rd),
                           "create_xfer_dlist(remote-pool)");
                for (int i = 0; i < remote_ring; ++i) {
                    uintptr_t addr = remote_buf_addr + size_t(i) * args.desc_len;
                    CHECK_NIXL(nixl_capi_xfer_dlist_add_desc(rd, addr,
                                                              args.pool_sizes[k], mem_dev_id),
                               "xfer_dlist_add_desc(remote-pool)");
                }
                CHECK_NIXL(nixl_capi_prep_xfer_dlist(agent,
                                                      conns[slot].remote_name.c_str(),
                                                      rd,
                                                      &conns[slot].remote_pool_h[k],
                                                      backend_opts),
                           "prep_xfer_dlist(remote-pool)");
                // Don't destroy rd: we keep it alive for the lifetime of the prepped handle.
                // (NIXL may or may not retain a reference; safest to keep both.)
            }
        }

        std::cerr << "rank " << args.rank << " conn-slot " << slot
                  << ": connected to '" << conns[slot].remote_name << "'\n";
    }

    if (args.rank == 1) {
        if (args.rank1_inflight > 0) {
            std::cerr << "rank 1 slot " << my_peer_idx
                      << ": starting active send-back loop, inflight="
                      << args.rank1_inflight << "\n";
            std::atomic<bool> stop{false};
            std::thread tx([&]() {
                Connection& c = conns[0];
                std::deque<nixl_capi_xfer_req_t> q;
                int cur = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    int li = cur % args.recv_ring_size;
                    int ri = (cur + 1) % args.send_ring_size;
                    cur++;
                    int loc[1] = {li};
                    int rem[1] = {ri};
                    nixl_capi_xfer_req_t req = nullptr;
                    if (nixl_capi_make_xfer_req(agent, NIXL_CAPI_XFER_OP_WRITE,
                            local_dlist_h, loc, 1,
                            c.remote_dlist_h, rem, 1,
                            &req, backend_opts) != NIXL_CAPI_SUCCESS) continue;
                    nixl_capi_status_t ps = nixl_capi_post_xfer_req(
                            agent, req, backend_opts);
                    if (ps != NIXL_CAPI_SUCCESS && ps != NIXL_CAPI_IN_PROG) {
                        nixl_capi_release_xfer_req(agent, req);
                        continue;
                    }
                    q.push_back(req);
                    while ((int)q.size() >= args.rank1_inflight) {
                        nixl_capi_xfer_req_t h = q.front();
                        for (;;) {
                            nixl_capi_status_t s = nixl_capi_get_xfer_status(agent, h);
                            if (s == NIXL_CAPI_SUCCESS) break;
                            if (s != NIXL_CAPI_IN_PROG) break;
                        }
                        nixl_capi_release_xfer_req(agent, h);
                        q.pop_front();
                    }
                }
            });
            for (;;) std::this_thread::sleep_for(std::chrono::seconds(60));
        }
        for (;;) std::this_thread::sleep_for(std::chrono::seconds(60));
    }

    std::cerr << "rank 0: ready, num_peers=" << args.num_peers
              << " inflight=" << args.inflight
              << " iters=" << args.iters
              << " fix_mode=" << fix_mode_str(args.fix_mode)
              << " bytes_list_size=" << args.bytes_list.size() << "\n";

    struct InflightSlot {
        nixl_capi_xfer_req_t req = nullptr;
        int conn_idx = -1;
        // For per-call mode: raw dlists to destroy after the request completes.
        nixl_capi_xfer_dlist_t pc_local = nullptr;
        nixl_capi_xfer_dlist_t pc_remote = nullptr;
    };
    std::deque<InflightSlot> inflight_q;

    std::vector<uint64_t> make_lat;
    std::vector<uint64_t> post_lat;
    make_lat.reserve(args.iters);
    post_lat.reserve(args.iters);

    auto drain_one = [&]() {
        if (inflight_q.empty()) return;
        InflightSlot& head = inflight_q.front();
        for (;;) {
            nixl_capi_status_t s = nixl_capi_get_xfer_status(agent, head.req);
            if (s == NIXL_CAPI_SUCCESS) break;
            if (s != NIXL_CAPI_IN_PROG) {
                std::cerr << "FATAL nixl: get_xfer_status err " << s << "\n";
                std::exit(4);
            }
        }
        CHECK_NIXL(nixl_capi_release_xfer_req(agent, head.req),
                   "release_xfer_req");
        if (head.pc_local) nixl_capi_destroy_xfer_dlist(head.pc_local);
        if (head.pc_remote) nixl_capi_destroy_xfer_dlist(head.pc_remote);
        inflight_q.pop_front();
    };

    int total_iters = args.warmup + args.iters;
    int desc_cursor = 0;
    int bytes_cursor = 0;
    uint64_t wall0 = now_ns();
    for (int i = 0; i < total_iters; ++i) {
        int conn_idx = i % n_conn;
        Connection& c = conns[conn_idx];
        int li = (desc_cursor) % args.send_ring_size;
        int ri = (desc_cursor + 1) % args.recv_ring_size;
        desc_cursor++;
        size_t bytes_to_write = args.bytes_list[bytes_cursor % args.bytes_list.size()];
        bytes_cursor++;

        nixl_capi_xfer_req_t req = nullptr;
        InflightSlot slot;
        slot.conn_idx = conn_idx;
        uint64_t t0 = 0, t1 = 0, t2 = 0;

        if (args.fix_mode == FixMode::CURRENT || args.fix_mode == FixMode::TIGHT_FIXED) {
            // Baseline path: prepped handles, fixed desc_len descriptors.
            int loc[1] = {li};
            int rem[1] = {ri};
            t0 = now_ns();
            CHECK_NIXL(nixl_capi_make_xfer_req(agent, NIXL_CAPI_XFER_OP_WRITE,
                                               local_dlist_h, loc, 1,
                                               c.remote_dlist_h, rem, 1,
                                               &req, backend_opts),
                       "make_xfer_req(current)");
            t1 = now_ns();
            nixl_capi_status_t ps = nixl_capi_post_xfer_req(agent, req, backend_opts);
            t2 = now_ns();
            if (ps != NIXL_CAPI_SUCCESS && ps != NIXL_CAPI_IN_PROG) {
                std::cerr << "FATAL: post_xfer_req status=" << ps << "\n";
                std::exit(3);
            }
        } else if (args.fix_mode == FixMode::LOG2_POOL) {
            int k = pick_pool(args.pool_sizes, bytes_to_write);
            int loc[1] = {li};
            int rem[1] = {ri};
            t0 = now_ns();
            CHECK_NIXL(nixl_capi_make_xfer_req(agent, NIXL_CAPI_XFER_OP_WRITE,
                                               local_pool_h[k], loc, 1,
                                               c.remote_pool_h[k], rem, 1,
                                               &req, backend_opts),
                       "make_xfer_req(log2-pool)");
            t1 = now_ns();
            nixl_capi_status_t ps = nixl_capi_post_xfer_req(agent, req, backend_opts);
            t2 = now_ns();
            if (ps != NIXL_CAPI_SUCCESS && ps != NIXL_CAPI_IN_PROG) {
                std::cerr << "FATAL: post_xfer_req status=" << ps << "\n";
                std::exit(3);
            }
        } else /* PER_CALL */ {
            // Build fresh raw 1-elem dlists sized to bytes_to_write, then use
            // create_xfer_req (no prep step required by NIXL).
            uintptr_t local_addr = (uintptr_t)buf + size_t(li) * args.desc_len;
            uintptr_t remote_addr = c.remote_buf_addr + size_t(ri) * args.desc_len;
            t0 = now_ns();
            nixl_capi_xfer_dlist_t l = nullptr;
            nixl_capi_xfer_dlist_t r = nullptr;
            CHECK_NIXL(nixl_capi_create_xfer_dlist(mem_type, &l),
                       "create_xfer_dlist(per-call-local)");
            CHECK_NIXL(nixl_capi_xfer_dlist_add_desc(l, local_addr, bytes_to_write, mem_dev_id),
                       "add_desc(per-call-local)");
            CHECK_NIXL(nixl_capi_create_xfer_dlist(mem_type, &r),
                       "create_xfer_dlist(per-call-remote)");
            CHECK_NIXL(nixl_capi_xfer_dlist_add_desc(r, remote_addr, bytes_to_write, mem_dev_id),
                       "add_desc(per-call-remote)");
            CHECK_NIXL(nixl_capi_create_xfer_req(agent, NIXL_CAPI_XFER_OP_WRITE,
                                                  l, r, c.remote_name.c_str(),
                                                  &req, backend_opts),
                       "create_xfer_req(per-call)");
            t1 = now_ns();
            nixl_capi_status_t ps = nixl_capi_post_xfer_req(agent, req, backend_opts);
            t2 = now_ns();
            if (ps != NIXL_CAPI_SUCCESS && ps != NIXL_CAPI_IN_PROG) {
                std::cerr << "FATAL: post_xfer_req status=" << ps << "\n";
                std::exit(3);
            }
            slot.pc_local = l;
            slot.pc_remote = r;
        }

        slot.req = req;
        inflight_q.push_back(slot);
        if ((int)inflight_q.size() >= args.inflight) {
            drain_one();
        }

        if (i >= args.warmup) {
            make_lat.push_back(t1 - t0);
            post_lat.push_back(t2 - t1);
        }
    }
    while (!inflight_q.empty()) drain_one();
    uint64_t wall1 = now_ns();

    double p50_m = pct(make_lat, 0.50);
    double p90_m = pct(make_lat, 0.90);
    double p99_m = pct(make_lat, 0.99);
    double pmax_m = make_lat.empty() ? 0.0 : double(make_lat.back());
    double p50_p = pct(post_lat, 0.50);
    double p90_p = pct(post_lat, 0.90);
    double p99_p = pct(post_lat, 0.99);
    double pmax_p = post_lat.empty() ? 0.0 : double(post_lat.back());

    // bytes_list summary for the result line
    std::string bytes_str;
    for (size_t i = 0; i < args.bytes_list.size(); ++i) {
        if (i) bytes_str += "+";
        bytes_str += std::to_string(args.bytes_list[i]);
    }

    std::cout << "RESULT,model=shared_agent_mesh"
              << ",fix_mode=" << fix_mode_str(args.fix_mode)
              << ",desc_len=" << args.desc_len
              << ",bytes_to_write=" << bytes_str
              << ",num_peers=" << args.num_peers
              << ",inflight=" << args.inflight
              << ",num_workers=" << args.num_workers
              << ",iters=" << args.iters
              << ",mem=" << args.mem
              << ",total_samples=" << make_lat.size()
              << ",make_p50_ns=" << p50_m
              << ",make_p90_ns=" << p90_m
              << ",make_p99_ns=" << p99_m
              << ",make_max_ns=" << pmax_m
              << ",post_p50_ns=" << p50_p
              << ",post_p90_ns=" << p90_p
              << ",post_p99_ns=" << p99_p
              << ",post_max_ns=" << pmax_p
              << ",wall_total_s=" << double(wall1 - wall0) / 1e9
              << ",throughput_per_s="
              << double(make_lat.size()) / (double(wall1 - wall0) / 1e9)
              << "\n";

    if (!args.out_csv.empty()) {
        std::ofstream csv(args.out_csv);
        csv << "iter,make_ns,post_ns\n";
        for (size_t i = 0; i < make_lat.size(); ++i) {
            csv << i << "," << make_lat[i] << "," << post_lat[i] << "\n";
        }
    }
    return 0;
}
