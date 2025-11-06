#pragma once

#include <string>
#include <exception>
#include <stdexcept>

const int ZMQ_PORT_BASE = 24000;
const int ZMQ_CPU_PORT_BASE = 35000;
const int ZMQ_GROUP_PORT = 46000;
const int ZMQ_MAGIC_MOD = 1007;
const int ZMQ_OFFSET_BASE = 16;

const int UCXQ_CPU_PORT_BASE = 36000;
const int UCXQ_OFFSET_BASE = 16;

#ifndef N_EXPERTS
#define N_EXPERTS 8
#endif

#ifndef EOS_TOKEN_ID
#define EOS_TOKEN_ID 2
#endif

#ifndef TEMP_DIR
#define TEMP_DIR "/tmp/disagmoe/"
#endif

#ifndef GROUP_CHANNEL_BUFFER_SIZE
#define GROUP_CHANNEL_BUFFER_SIZE 8192
#endif

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 512
#endif

#ifndef MAX_N_EXPERTS
#define MAX_N_EXPERTS 8

#endif

// Limit for number of pending receives drained per MuPool::run() iteration
// We need to have such a limit to prevent livelock
#ifndef MU_POOL_GROUP_RECV_LIMIT
#define MU_POOL_GROUP_RECV_LIMIT 16
#endif

#define ASSERT(condition) do {if (!(condition)) { \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " Assertion failed: " + std::string(#condition)); \
}} while(0)

#define ASSERT_MSG(condition, msg) do {if (!(condition)) { \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
    " Assertion failed: " + std::string(#condition) + ", message: " + msg); \
}} while(0)