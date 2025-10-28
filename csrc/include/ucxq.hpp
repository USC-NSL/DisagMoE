/*
    A simple UCX-based MQ implementation. Both sender and receiver sockets are for single thread use.
    Optimized for fan-in pattern. API is similar to ZMQ.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cstring>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>

#include <ucp/api/ucp.h>

namespace ucxq {

using recv_result_t = std::optional<size_t>;

struct send_flags {
    enum type : int {
        none = 0,
        sndmore = 1,
    };
};

// Buffer view helper, just likezmq::buffer/str_buffer
struct buffer_view {
    const void* data;
    size_t size;
};

inline buffer_view buffer(const void* ptr, size_t len) { return buffer_view{ptr, len}; }
inline buffer_view str_buffer(const char* cstr) { return buffer_view{cstr, std::strlen(cstr) + 1}; }

// Message container
class message_t {
public:
    explicit message_t(size_t size = 0) : storage_(size) {}
    void* data() { return storage_.data(); }
    const void* data() const { return storage_.data(); }
    size_t size() const { return storage_.size(); }
    void resize(size_t n) { storage_.resize(n); }
    void assign(std::vector<uint8_t>&& bytes) { storage_ = std::move(bytes); }
    void assign(const void* src, size_t len) {
        storage_.resize(len);
        if (len) std::memcpy(storage_.data(), src, len);
    }
    std::string to_string() const { return std::string(reinterpret_cast<const char*>(storage_.data()), storage_.size()); }

private:
    std::vector<uint8_t> storage_;
};

struct Message {
    std::vector<uint8_t> data;
};

// Single-producer single-consumer ring buffer.
// In our case, the producer is UCX worker callbacks, the consumer is the receiver thread.
// Capacity must be power of two.
template <typename T>
class SpscRing {
public:
    explicit SpscRing(size_t capacity_pow2)
        : capacity_mask_(capacity_pow2 - 1),
          buffer_(capacity_pow2),
          head_(0),
          tail_(0) {}

    bool try_enqueue(T&& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & capacity_mask_;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false; // full
        }
        buffer_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool try_dequeue(T& out) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false; // empty
        }
        out = std::move(buffer_[tail]);
        tail_.store((tail + 1) & capacity_mask_, std::memory_order_release);
        return true;
    }

private:
    size_t capacity_mask_;
    std::vector<T> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

class FanInQueueReceiver {
public:
    FanInQueueReceiver(const std::string& ip, int tcp_port);
    ~FanInQueueReceiver();

    void start();
    // Start a TCP server to accept UCX address exchanges from remote senders
    void start_server();
    // Add another remote sender endpoint to this receiver (fan-in)
    void add_remote(const std::string& ip, int tcp_port);
    bool dequeue(Message& out, bool is_non_blocking = false);
    size_t endpoint_count() const;
    void stop();

private:
    void progressThread();
    static void onRecvCallback(void* request, ucs_status_t status, const ucp_tag_recv_info_t* info, void* user_data);

    int tcp_port_ = 0;
    std::string ip_;
    int tcp_listen_fd_ = -1;

    ucp_context_h context_ = nullptr;
    ucp_worker_h worker_ = nullptr;

    std::vector<ucp_ep_h> eps_;
    std::atomic<size_t> ep_count_{0};
    std::mutex eps_mu_;

    std::thread progress_thr_;
    std::thread accept_thr_;
    std::atomic<bool> running_{false};

    SpscRing<Message> q_{1024};
};

class FanInQueueSender {
public:
    FanInQueueSender(const std::string& ip, int tcp_port);
    ~FanInQueueSender();

    // Start a TCP server for receiver-connect/sender-bind semantics
    void start_server();
    // Connect to a receiver that is listening (receiver-bind / sender-connect)
    void add_remote(const std::string& ip, int tcp_port);
    void send(const void* buf, size_t len, uint64_t tag = 0xABCDEF);
    size_t endpoint_count() const;
    void stop();

private:
    void waitReq(void* req);

    int tcp_listen_fd_ = -1;
    int tcp_port_ = 0;
    std::string ip_;

    ucp_context_h context_ = nullptr;
    ucp_worker_h worker_ = nullptr;
    ucp_ep_h ep_ = nullptr;
};

enum class socket_type { push, pull };

// ZMQ-like socket that maps PUSH to sender and PULL to receiver.
class socket_t {
public:
    explicit socket_t(socket_type type);
    ~socket_t();

    // Bind for senders (push). Endpoint format: tcp://IP:PORT
    void bind(const std::string& endpoint);

    // Connect for receivers (pull). Endpoint format: tcp://IP:PORT
    void connect(const std::string& endpoint);

    // Send API (single frame or multipart where caller indicates sndmore for all but last)
    bool send(buffer_view buf, send_flags::type flags = send_flags::none);

    // Receive single frame (used for tensor data path)
    recv_result_t recv(message_t& msg, bool is_non_blocking = false);

    // Receive multipart (peer_id + metadata). Returns number of frames.
    recv_result_t recv_multipart(std::vector<message_t>& out_frames, bool is_non_blocking = false);

private:
    // Internal helpers
    static void parse_tcp_endpoint(const std::string& endpoint, std::string& ip, int& port);
    static constexpr uint32_t kMultipartMagic = 0x55435851; // 'UCXQ'

    // Multipart staging for send()
    std::vector<uint8_t> multipart_staging_;
    bool multipart_open_ = false;
    uint32_t multipart_frame_count_ = 0;

    socket_type type_;
    enum class init_mode { none, bind, connect };
    init_mode mode_ = init_mode::none; // guard: allow only one of bind() or connect()
    // UCX queue components
    std::unique_ptr<FanInQueueReceiver> receiver_;
    std::unique_ptr<FanInQueueSender> sender_;
};

} // namespace ucxq
