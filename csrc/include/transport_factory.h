#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "comm.h"

namespace disagmoe {

// Select transport backend at runtime. Must be called from Python before init.
// name: "ucx" or "zmq"
void select_transport(const std::string &name);

// ----- 2. MQ socket abstraction (wraps ucxq::socket_t/zmq::socket_t) -----
class MqSocket {
public:
    virtual ~MqSocket() = default;
    virtual void bind(const std::string &endpoint) = 0;
    virtual void connect(const std::string &endpoint) = 0;
    virtual void send_multipart(const std::string &frame0, const void *data, size_t size) = 0;
    virtual bool recv_multipart(std::string &frame0, std::vector<uint8_t> &frame1, bool non_blocking = false) = 0;
    virtual void send(const void *data, size_t size) = 0;
    virtual bool recv(std::vector<uint8_t> &data, bool non_blocking = false) = 0;
};

using MqSocketPtr = std::unique_ptr<MqSocket>;
using MqFactory = std::function<MqSocketPtr(bool /*isPush*/)>;

// Create MQ sockets of the selected backend
const MqFactory &mq_factory();

// Endpoint address factory for the selected backend
using EndpointFactory = std::function<std::string(int /*device_id*/, bool /*is_gpu*/, int /*manual_port*/)>;
const EndpointFactory &mq_endpoint_factory();

} // namespace disagmoe



