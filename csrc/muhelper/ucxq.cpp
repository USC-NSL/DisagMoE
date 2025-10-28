/*
    A simple UCX-based MQ implementation. Both sender and receiver sockets are for single thread use.
    Optimized for fan-in pattern. API is similar to ZMQ.
*/

#include "ucxq.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cassert>

namespace ucxq {

namespace {

static void append_u32(std::vector<uint8_t>& v, uint32_t x) {
    uint32_t be = htonl(x);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&be);
    v.insert(v.end(), p, p + sizeof(be));
}

static uint32_t read_u32(const uint8_t* p) {
    uint32_t be;
    std::memcpy(&be, p, sizeof(be));
    return ntohl(be);
}

static void store_u32(std::vector<uint8_t>& v, size_t offset, uint32_t value) {
    uint32_t be = htonl(value);
    std::memcpy(v.data() + offset, &be, sizeof(be));
}

struct RecvCtx {
    FanInQueueReceiver* self;
    std::vector<uint8_t>* buf;
};
}

FanInQueueReceiver::FanInQueueReceiver(const std::string& ip, int tcp_port)
    : tcp_port_(tcp_port), ip_(ip) {
    ucp_config_t* cfg{};
    if (ucp_config_read(nullptr, nullptr, &cfg) != UCS_OK) {
        throw std::runtime_error("ucp_config_read failed");
    }
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_TAG;
    if (ucp_init(&params, cfg, &context_) != UCS_OK) {
        ucp_config_release(cfg);
        throw std::runtime_error("ucp_init failed");
    }
    ucp_config_release(cfg);

    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    if (ucp_worker_create(context_, &worker_params, &worker_) != UCS_OK) {
        ucp_cleanup(context_);
        context_ = nullptr;
        throw std::runtime_error("ucp_worker_create failed");
    }
}

FanInQueueReceiver::~FanInQueueReceiver() { stop(); }

void FanInQueueReceiver::start() {
    running_.store(true, std::memory_order_release);
    // Start UCX worker progress thread to poll for incoming messages
    progress_thr_ = std::thread(&FanInQueueReceiver::progressThread, this);
}

void FanInQueueReceiver::start_server() {
    running_.store(true, std::memory_order_release);
    // Start UCX worker progress/recv thread first
    progress_thr_ = std::thread(&FanInQueueReceiver::progressThread, this);

    // Listen as a server for address exchange with senders
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port_);
    if (inet_pton(AF_INET, ip_.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        throw std::runtime_error("inet_pton failed");
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        throw std::runtime_error("bind failed");
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        throw std::runtime_error("listen failed");
    }

    tcp_listen_fd_ = fd;
    accept_thr_ = std::thread([this]() {
        while (running_.load(std::memory_order_acquire)) {
            int cfd = ::accept(tcp_listen_fd_, nullptr, nullptr);
            if (cfd < 0) {
                if (!running_.load(std::memory_order_acquire)) break;
                // transient error, continue
                std::this_thread::yield();
                continue;
            }

            // First, receive peer UCX address length and bytes
            uint32_t peer_len_be{};
            if (recv(cfd, &peer_len_be, sizeof(peer_len_be), MSG_WAITALL) != static_cast<ssize_t>(sizeof(peer_len_be))) {
                close(cfd);
                continue;
            }
            uint32_t peer_len = ntohl(peer_len_be);
            std::vector<uint8_t> peer(peer_len);
            if (recv(cfd, peer.data(), peer.size(), MSG_WAITALL) != static_cast<ssize_t>(peer.size())) {
                close(cfd);
                continue;
            }

            // Then send my UCX worker address
            ucp_address_t* my_addr{};
            size_t my_len{};
            if (ucp_worker_get_address(worker_, &my_addr, &my_len) != UCS_OK) {
                close(cfd);
                continue;
            }
            uint32_t len_be = htonl(static_cast<uint32_t>(my_len));
            ::send(cfd, &len_be, sizeof(len_be), 0);
            ::send(cfd, my_addr, my_len, 0);
            ucp_worker_release_address(worker_, my_addr);

            // Create EP
            ucp_ep_params_t ep_params{};
            ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
            ep_params.address = reinterpret_cast<ucp_address_t*>(peer.data());
            ucp_ep_h ep{};
            if (ucp_ep_create(worker_, &ep_params, &ep) == UCS_OK) {
                {
                    std::lock_guard<std::mutex> lock(eps_mu_);
                    eps_.push_back(ep);
                }
                ep_count_.fetch_add(1, std::memory_order_relaxed);
            } else {
                throw std::runtime_error("add_remote failed: ep create failed");
            }

            close(cfd);
        }
    });
}

void FanInQueueReceiver::add_remote(const std::string& ip, int tcp_port) {
    // Connect to a remote sender to exchange UCX addresses and create an EP
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        throw std::runtime_error("inet_pton failed");
    }
    int attempts = 0;
    constexpr int kMaxAttempts = 400;
    while (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (++attempts >= kMaxAttempts) {
            close(fd);
            throw std::runtime_error("connect failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    ucp_address_t* my_addr{};
    size_t my_len{};
    if (ucp_worker_get_address(worker_, &my_addr, &my_len) != UCS_OK) {
        close(fd);
        throw std::runtime_error("get_address failed");
    }
    uint32_t len_be = htonl(static_cast<uint32_t>(my_len));
    ::send(fd, &len_be, sizeof(len_be), 0);
    ::send(fd, my_addr, my_len, 0);
    ucp_worker_release_address(worker_, my_addr);

    uint32_t peer_len_be{};
    if (recv(fd, &peer_len_be, sizeof(peer_len_be), MSG_WAITALL) != static_cast<ssize_t>(sizeof(peer_len_be))) {
        close(fd);
        throw std::runtime_error("recv len failed");
    }
    uint32_t peer_len = ntohl(peer_len_be);
    std::vector<uint8_t> peer(peer_len);
    if (recv(fd, peer.data(), peer.size(), MSG_WAITALL) != static_cast<ssize_t>(peer.size())) {
        close(fd);
        throw std::runtime_error("recv addr failed");
    }

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer.data());
    ucp_ep_h ep{};
    if (ucp_ep_create(worker_, &ep_params, &ep) == UCS_OK) {
        {
            std::lock_guard<std::mutex> lock(eps_mu_);
            eps_.push_back(ep);
        }
        ep_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
        throw std::runtime_error("add_remote failed: ep create failed");
    }
    close(fd);
}


void FanInQueueReceiver::progressThread() {
    const uint64_t TAG = 0xABCDEF;
    while (running_.load(std::memory_order_acquire)) {
        int posted = 0;
        for (; posted < 64 && running_.load(std::memory_order_acquire); ++posted) {
            ucp_tag_recv_info_t info{};
            ucp_tag_message_h msg = ucp_tag_probe_nb(worker_, TAG, static_cast<uint64_t>(-1), 1, &info);
            if (msg == nullptr) {
                break;
            }

            auto* buf = new std::vector<uint8_t>(info.length);
            auto* ctx = new RecvCtx{this, buf};

            ucp_request_param_t param{};
            param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
            param.user_data = ctx;
            param.cb.recv = &FanInQueueReceiver::onRecvCallback;

            void* req = ucp_tag_msg_recv_nbx(worker_, buf->data(), buf->size(), msg, &param);
            if (UCS_PTR_IS_ERR(req)) {
                delete buf;
                delete ctx;
                continue;
            }
            if (!UCS_PTR_IS_PTR(req)) {
                ucs_status_t status = static_cast<ucs_status_t>(UCS_PTR_STATUS(req));
                if (status == UCS_OK) {
                    Message m;
                    m.data = std::move(*buf);
                    bool enqueued = false;
                    for (int spin = 0; spin < 64; ++spin) {
                        if (q_.try_enqueue(std::move(m))) {
                            enqueued = true;
                            break;
                        }
                        std::this_thread::yield();
                    }
                    assert(enqueued && "ucxq FanInQueueReceiver queue overflow (progressThread)");
                }
                delete buf;
                delete ctx;
            }
        }

        ucp_worker_progress(worker_); // drive the worker
    }
}

void FanInQueueReceiver::onRecvCallback(void* request, ucs_status_t status, const ucp_tag_recv_info_t*, void* user_data) {
    auto* ctx = static_cast<RecvCtx*>(user_data);
    FanInQueueReceiver* self = ctx->self;
    std::vector<uint8_t>* buf = ctx->buf;

    if (status == UCS_OK) {
        Message m;
        m.data = std::move(*buf);
        bool enqueued = false;
        for (int spin = 0; spin < 64; ++spin) {
            if (self->q_.try_enqueue(std::move(m))) {
                enqueued = true;
                break;
            }
            std::this_thread::yield();
        }
        assert(enqueued && "ucxq FanInQueueReceiver queue overflow (onRecvCallback)");
    }

    delete buf;
    delete ctx;
    if (UCS_PTR_IS_PTR(request)) {
        ucp_request_free(request);
    }
}

bool FanInQueueReceiver::dequeue(Message& out, bool is_non_blocking) {
    if (is_non_blocking) {
        return q_.try_dequeue(out);
    }
    while (running_.load(std::memory_order_acquire)) {
        if (q_.try_dequeue(out)) {
            return true;
        }
    }
}

void FanInQueueReceiver::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if (progress_thr_.joinable()) {
        progress_thr_.join();
    }
    if (tcp_listen_fd_ >= 0) {
        close(tcp_listen_fd_);
        tcp_listen_fd_ = -1;
    }
    if (accept_thr_.joinable()) {
        accept_thr_.join();
    }
    {
        std::lock_guard<std::mutex> lock(eps_mu_);
        for (auto ep : eps_) {
            ucp_ep_destroy(ep);
        }
        eps_.clear();
    }
    if (worker_) {
        ucp_worker_destroy(worker_);
        worker_ = nullptr;
    }
    if (context_) {
        ucp_cleanup(context_);
        context_ = nullptr;
    }
}

size_t FanInQueueReceiver::endpoint_count() const {
    return ep_count_.load(std::memory_order_relaxed);
}

FanInQueueSender::FanInQueueSender(const std::string& ip, int tcp_port)
    : tcp_port_(tcp_port), ip_(ip) {
    ucp_config_t* cfg{};
    if (ucp_config_read(nullptr, nullptr, &cfg) != UCS_OK) {
        throw std::runtime_error("ucp_config_read failed");
    }
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_TAG;
    if (ucp_init(&params, cfg, &context_) != UCS_OK) {
        ucp_config_release(cfg);
        throw std::runtime_error("ucp_init failed");
    }
    ucp_config_release(cfg);

    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    if (ucp_worker_create(context_, &worker_params, &worker_) != UCS_OK) {
        ucp_cleanup(context_);
        context_ = nullptr;
        throw std::runtime_error("ucp_worker_create failed");
    }
}

FanInQueueSender::~FanInQueueSender() { stop(); }

void FanInQueueSender::start_server() {
    // Listen as a server for address exchange
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port_);
    if (inet_pton(AF_INET, ip_.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        throw std::runtime_error("inet_pton failed");
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        throw std::runtime_error("bind failed");
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        throw std::runtime_error("listen failed");
    }

    int cfd = ::accept(fd, nullptr, nullptr);
    if (cfd < 0) {
        close(fd);
        throw std::runtime_error("accept failed");
    }

    // First, receive peer UCX address length and bytes
    uint32_t peer_len_be{};
    if (recv(cfd, &peer_len_be, sizeof(peer_len_be), MSG_WAITALL) != static_cast<ssize_t>(sizeof(peer_len_be))) {
        close(cfd);
        close(fd);
        throw std::runtime_error("recv len failed");
    }
    uint32_t peer_len = ntohl(peer_len_be);
    std::vector<uint8_t> peer(peer_len);
    if (recv(cfd, peer.data(), peer.size(), MSG_WAITALL) != static_cast<ssize_t>(peer.size())) {
        close(cfd);
        close(fd);
        throw std::runtime_error("recv addr failed");
    }

    // Then send my UCX worker address
    ucp_address_t* my_addr{};
    size_t my_len{};
    if (ucp_worker_get_address(worker_, &my_addr, &my_len) != UCS_OK) {
        close(cfd);
        close(fd);
        throw std::runtime_error("get_address failed");
    }
    uint32_t len_be = htonl(static_cast<uint32_t>(my_len));
    ::send(cfd, &len_be, sizeof(len_be), 0);
    ::send(cfd, my_addr, my_len, 0);
    ucp_worker_release_address(worker_, my_addr);

    // Create EP
    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer.data());
    if (ucp_ep_create(worker_, &ep_params, &ep_) != UCS_OK) {
        close(cfd);
        close(fd);
        throw std::runtime_error("ep create failed");
    }
    close(cfd);
    close(fd);
}

void FanInQueueSender::add_remote(const std::string& ip, int tcp_port) {
    if (ep_ != nullptr) {
        return;
    }
    // Connect as a client to the receiver server for UCX address exchange
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        throw std::runtime_error("inet_pton failed");
    }
    int attempts = 0;
    constexpr int kMaxAttempts = 400;
    while (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (++attempts >= kMaxAttempts) {
            close(fd);
            throw std::runtime_error("connect failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Send my UCX worker address, then receive peer's, and create an endpoint
    ucp_address_t* my_addr{};
    size_t my_len{};
    if (ucp_worker_get_address(worker_, &my_addr, &my_len) != UCS_OK) {
        close(fd);
        throw std::runtime_error("get_address failed");
    }
    uint32_t len_be = htonl(static_cast<uint32_t>(my_len));
    if (::send(fd, &len_be, sizeof(len_be), 0) != static_cast<ssize_t>(sizeof(len_be))) {
        ucp_worker_release_address(worker_, my_addr);
        close(fd);
        throw std::runtime_error("send len failed");
    }
    if (::send(fd, my_addr, my_len, 0) != static_cast<ssize_t>(my_len)) {
        ucp_worker_release_address(worker_, my_addr);
        close(fd);
        throw std::runtime_error("send addr failed");
    }
    ucp_worker_release_address(worker_, my_addr);

    uint32_t peer_len_be{};
    if (recv(fd, &peer_len_be, sizeof(peer_len_be), MSG_WAITALL) != static_cast<ssize_t>(sizeof(peer_len_be))) {
        close(fd);
        throw std::runtime_error("recv len failed");
    }
    uint32_t peer_len = ntohl(peer_len_be);
    std::vector<uint8_t> peer(peer_len);
    if (recv(fd, peer.data(), peer.size(), MSG_WAITALL) != static_cast<ssize_t>(peer.size())) {
        close(fd);
        throw std::runtime_error("recv addr failed");
    }

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer.data());
    if (ucp_ep_create(worker_, &ep_params, &ep_) != UCS_OK) {
        close(fd);
        throw std::runtime_error("ep create failed");
    }
    close(fd);
}

void FanInQueueSender::send(const void* buf, size_t len, uint64_t tag) {
    if (ep_ == nullptr) {
        return;
    }
    ucp_request_param_t param{};
    param.op_attr_mask = 0;
    void* req = ucp_tag_send_nbx(ep_, buf, len, tag, &param);
    waitReq(req);
}

void FanInQueueSender::waitReq(void* req) {
    if (req == nullptr) {
        return;
    }
    if (!UCS_PTR_IS_PTR(req)) {
        ucs_status_t status = static_cast<ucs_status_t>(UCS_PTR_STATUS(req));
        if (status != UCS_OK) {
            std::cerr << "UCX err: " << ucs_status_string(status) << std::endl;
            std::abort();
        }
        return;
    }
    while (ucp_request_check_status(req) == UCS_INPROGRESS) {
        ucp_worker_progress(worker_);
    }
    ucp_request_free(req);
}

void FanInQueueSender::stop() {
    if (ep_ != nullptr) {
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = UCP_EP_CLOSE_FLAG_FORCE;
        void* req = ucp_ep_close_nbx(ep_, &param);
        if (UCS_PTR_IS_PTR(req)) {
            while (ucp_request_check_status(req) == UCS_INPROGRESS) {
                ucp_worker_progress(worker_);
            }
            ucp_request_free(req);
        }
        ep_ = nullptr;
    }
    if (worker_) {
        ucp_worker_destroy(worker_);
        worker_ = nullptr;
    }
    if (context_) {
        ucp_cleanup(context_);
        context_ = nullptr;
    }
}

size_t FanInQueueSender::endpoint_count() const {
    return ep_ != nullptr ? 1 : 0;
}

socket_t::socket_t(socket_type type) : type_(type) {}

socket_t::~socket_t() {
    if (sender_) sender_->stop();
    if (receiver_) receiver_->stop();
}

void socket_t::parse_tcp_endpoint(const std::string& endpoint, std::string& ip, int& port) {
    // Expect format tcp://IP:PORT
    const auto scheme_pos = endpoint.find("://");
    if (scheme_pos == std::string::npos) {
        throw std::invalid_argument("Endpoint must contain scheme://");
    }
    const std::string scheme = endpoint.substr(0, scheme_pos);
    if (scheme != "tcp" && scheme != "ucx") {
        throw std::invalid_argument("Only tcp:// or ucx:// endpoints are supported");
    }
    auto rest = endpoint.substr(scheme_pos + 3);
    auto colon = rest.find(":");
    if (colon == std::string::npos) {
        throw std::invalid_argument("Endpoint must be tcp://IP:PORT");
    }
    ip = rest.substr(0, colon);
    port = std::stoi(rest.substr(colon + 1));
}

// Bind for senders (push). Sender listens for receiver connects to exchange addresses
void socket_t::bind(const std::string& endpoint) {
    if (mode_ != init_mode::none) {
        throw std::logic_error("socket already initialized (bind/connect)");
    }
    std::string ip; int port = 0;
    parse_tcp_endpoint(endpoint, ip, port);
    if (type_ == socket_type::push) {
        sender_ = std::make_unique<ucxq::FanInQueueSender>(ip, port);
        sender_->start_server(); // sender-bind
    } else {
        receiver_ = std::make_unique<ucxq::FanInQueueReceiver>(ip, port);
        receiver_->start_server(); // receiver-bind
    }
    mode_ = init_mode::bind;
}

// Connect for receivers (pull). Receiver connects to each sender endpoint to exchange addresses
void socket_t::connect(const std::string& endpoint) {
    if (mode_ == init_mode::bind) {
        throw std::logic_error("socket already initialized with bind()");
    }
    std::string ip; int port = 0;
    parse_tcp_endpoint(endpoint, ip, port);
    if (type_ == socket_type::pull) {
        if (!receiver_) {
            receiver_ = std::make_unique<ucxq::FanInQueueReceiver>(ip, port);
            receiver_->start();
        } else {
        }
        receiver_->add_remote(ip, port); // receiver-connect
    } else {
        if (sender_ == nullptr) {
            sender_ = std::make_unique<ucxq::FanInQueueSender>(ip, port);
        } else {
        }
        sender_->add_remote(ip, port); // sender-connect
    }
    mode_ = init_mode::connect;
}

bool socket_t::send(buffer_view buf, send_flags::type flags) {
    if (type_ != socket_type::push || !sender_) {
        throw std::logic_error("either wrong socket type or sender not initialized");
    }

    auto append_frame = [&](buffer_view frame) {
        append_u32(multipart_staging_, static_cast<uint32_t>(frame.size));
        if (frame.size && frame.data != nullptr) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(frame.data);
            multipart_staging_.insert(multipart_staging_.end(), p, p + frame.size);
        }
        ++multipart_frame_count_;
        store_u32(multipart_staging_, sizeof(uint32_t), multipart_frame_count_);
    };

    if (flags == send_flags::sndmore) {
        if (!multipart_open_) {
            multipart_staging_.clear();
            multipart_staging_.reserve(sizeof(uint32_t) * 2 + buf.size);
            append_u32(multipart_staging_, kMultipartMagic);
            append_u32(multipart_staging_, 0); // placeholder for num_frames
            multipart_frame_count_ = 0;
            multipart_open_ = true;
        }
        append_frame(buf);
        return true;
    }

    if (multipart_open_) {
        append_frame(buf);
        sender_->send(multipart_staging_.data(), multipart_staging_.size());
        multipart_staging_.clear();
        multipart_open_ = false;
        multipart_frame_count_ = 0;
        return true;
    }

    sender_->send(buf.data, buf.size);
    return true;
}

recv_result_t socket_t::recv(message_t& msg, bool is_non_blocking) {
    if (type_ != socket_type::pull || !receiver_) {
        throw std::logic_error("either wrong socket type or receiver queue not initialized");
    }
    ucxq::Message m;
    if (!receiver_->dequeue(m, is_non_blocking)) return std::nullopt;

    // Deliver payload as-is. Callers needing multipart semantics should use recv_multipart().
    const size_t bytes = m.data.size();
    msg.assign(std::move(m.data));
    return recv_result_t{bytes};
}

recv_result_t socket_t::recv_multipart(std::vector<message_t>& out_frames, bool is_non_blocking) {
    if (type_ != socket_type::pull || !receiver_) {
        throw std::logic_error("either wrong socket type or receiver queue not initialized");
    }
    ucxq::Message m;
    if (!receiver_->dequeue(m, is_non_blocking)) return std::nullopt;

    std::vector<uint8_t> payload = std::move(m.data);
    if (payload.size() < sizeof(uint32_t) * 2) {
        out_frames.clear();
        out_frames.emplace_back();
        out_frames.back().assign(std::move(payload));
        return recv_result_t{1};
    }

    const uint8_t* p = payload.data();
    const uint8_t* end = p + payload.size();
    uint32_t magic = read_u32(p);
    p += sizeof(uint32_t);
    if (magic != kMultipartMagic) {
        out_frames.clear();
        out_frames.emplace_back();
        out_frames.back().assign(std::move(payload));
        return recv_result_t{1};
    }

    uint32_t num_frames = read_u32(p);
    p += sizeof(uint32_t);

    out_frames.clear();
    out_frames.reserve(num_frames);
    for (uint32_t i = 0; i < num_frames; ++i) {
        if (p + sizeof(uint32_t) > end) break;
        uint32_t len = read_u32(p);
        p += sizeof(uint32_t);
        if (p + len > end) break;
        out_frames.emplace_back();
        out_frames.back().assign(p, len);
        p += len;
    }
    return recv_result_t{out_frames.size()};
}

} // namespace ucxq
