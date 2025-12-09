/*
    With these classes, we provide a single unified interface
    for both ZMQ or UCX
*/

#include "transport_factory.h"
#include "distributed.hpp"
#include "logging.h"
#include "utils.hpp"

#include <mutex>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <chrono>

static zmq::context_t& GlobalZmqContext() {
    static zmq::context_t ctx(4);
    return ctx;
}

namespace disagmoe {

static MqFactory g_mq_factory;
static EndpointFactory g_ep_factory;
static std::once_flag g_selected_once;
static bool g_selected = false;

namespace {

// Optional non-blocking ZMQ send with backoff for debugging stalls.
static bool g_zmq_dontwait = true;      // always use non-blocking send with backoff
static long g_zmq_send_warn_us = 5000;  // warn if we spin longer than this (microseconds)

class ZmqSocketAdapter final : public MqSocket {
public:
    ZmqSocketAdapter(bool isPush)
        : ctx_(GlobalZmqContext()), sock_(ctx_, isPush ? zmq::socket_type::push : zmq::socket_type::pull) {

    }

    void bind(const std::string &endpoint) override { sock_.bind(endpoint); }
    void connect(const std::string &endpoint) override { sock_.connect(endpoint); }
    void send_multipart(const std::string &frame0, const void *data, size_t size) override {
        send_frame(zmq::buffer(frame0.data(), frame0.size()), zmq::send_flags::sndmore);
        send_frame(zmq::buffer(data, size), zmq::send_flags::none);
    }
    bool recv_multipart(std::string &frame0, std::vector<uint8_t> &frame1, bool non_blocking = false) override {
        zmq::message_t f0;
        zmq::message_t f1;
        auto flags = non_blocking ? zmq::recv_flags::dontwait : zmq::recv_flags::none;
        auto r0 = sock_.recv(f0, flags);
        if (!r0.has_value()) return false;
        auto r1 = sock_.recv(f1, zmq::recv_flags::none);
        if (!r1.has_value()) return false;
        frame0 = f0.to_string();
        frame1.resize(f1.size());
        if (f1.size()) std::memcpy(frame1.data(), f1.data(), f1.size());
        return true;
    }
    void send(const void *data, size_t size) override {
        send_frame(zmq::buffer(data, size), zmq::send_flags::none);
    }
    bool recv(std::vector<uint8_t> &data, bool non_blocking = false) override {
        zmq::message_t msg;
        auto flags = non_blocking ? zmq::recv_flags::dontwait : zmq::recv_flags::none;
        auto r = sock_.recv(msg, flags);
        if (!r.has_value()) return false;
        data.resize(msg.size());
        if (msg.size()) std::memcpy(data.data(), msg.data(), msg.size());
        return true;
    }

private:
    template <class Buffer>
    void send_frame(const Buffer& buf, zmq::send_flags base_flags) {
        if (!g_zmq_dontwait) {
            auto t0 = t_now();
            sock_.send(buf, base_flags);
            auto dur = static_cast<long long>(t_now()) - static_cast<long long>(t0);
            if (g_zmq_send_warn_us > 0 && dur > g_zmq_send_warn_us) {
                DMOE_LOG(WARNING) << "[ZMQ] blocking send " << dur << "us" << LEND;
            }
            return;
        }

        auto start = t_now();
        bool warned = false;
        while (true) {
            auto res = sock_.send(buf, base_flags | zmq::send_flags::dontwait);
            if (res.has_value()) return;
            auto waited = static_cast<long long>(t_now()) - static_cast<long long>(start);
            if (!warned && g_zmq_send_warn_us > 0 && waited > g_zmq_send_warn_us) {
                DMOE_LOG(WARNING) << "[ZMQ] send stalled " << waited << "us" << LEND;
                warned = true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    zmq::context_t &ctx_;
    zmq::socket_t sock_;
};

class UcxqSocketAdapter final : public MqSocket {
public:
    UcxqSocketAdapter(bool isPush)
        : sock_(std::make_unique<ucxq::socket_t>(isPush ? ucxq::socket_type::push : ucxq::socket_type::pull)) {}

    void bind(const std::string &endpoint) override { sock_->bind(endpoint); }
    void connect(const std::string &endpoint) override { sock_->connect(endpoint); }
    void send_multipart(const std::string &frame0, const void *data, size_t size) override {
        sock_->send(ucxq::str_buffer(frame0.c_str()), ucxq::send_flags::sndmore);
        sock_->send(ucxq::buffer(data, size));
    }
    bool recv_multipart(std::string &frame0, std::vector<uint8_t> &frame1, bool non_blocking = false) override {
        std::vector<ucxq::message_t> frames;
        auto res = sock_->recv_multipart(frames, non_blocking);
        if (!res.has_value() || *res != 2) return false;
        frame0 = frames[0].to_string();
        frame1.resize(frames[1].size());
        std::memcpy(frame1.data(), frames[1].data(), frames[1].size());
        return true;
    }
    void send(const void *data, size_t size) override {
        sock_->send(ucxq::buffer(data, size));
    }
    bool recv(std::vector<uint8_t> &data, bool non_blocking = false) override {
        ucxq::message_t msg;
        auto res = sock_->recv(msg, non_blocking);
        if (!res.has_value()) return false;
        data.resize(msg.size());
        if (msg.size()) std::memcpy(data.data(), msg.data(), msg.size());
        return true;
    }

private:
    std::unique_ptr<ucxq::socket_t> sock_;
};

} // namespace

void select_transport(const std::string &name) {
    if (name == "ucx") {
        g_mq_factory = [](bool isPush) { return std::make_unique<UcxqSocketAdapter>(isPush); };
        g_ep_factory = [](int device_id, bool is_gpu, int manual_port) {
            return get_ucxq_addr(device_id, is_gpu, manual_port);
        };
    } else if (name == "zmq") {
        g_mq_factory = [](bool isPush) { return std::make_unique<ZmqSocketAdapter>(isPush); };
        g_ep_factory = [](int device_id, bool is_gpu, int manual_port) {
            return get_zmq_addr(device_id, is_gpu, manual_port);
        };
    } else {
        throw std::runtime_error("Unknown transport: " + name);
    }
    g_selected = true;
}

const MqFactory &mq_factory() {
    if (!g_selected) {
        throw std::logic_error("select_transport(name) must be called before using factories");
    }
    return g_mq_factory;
}

const EndpointFactory &mq_endpoint_factory() {
    if (!g_selected) {
        throw std::logic_error("select_transport(name) must be called before using factories");
    }
    return g_ep_factory;
}

} // namespace disagmoe
