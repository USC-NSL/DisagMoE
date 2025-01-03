#pragma once

#include <string>
#include <exception>
#include <stdexcept>

const int ZMQ_PORT_BASE = 24000;
const int ZMQ_CPU_PORT_BASE = 35000;
const int ZMQ_GROUP_PORT = 46000;
const int ZMQ_MAGIC_MOD = 1007;
const int ZMQ_OFFSET_BASE = 16;

#ifndef N_EXPERTS
#define N_EXPERTS 8
#endif

#ifndef EOS_TOKEN_ID
#define EOS_TOKEN_ID 2
#endif

#ifndef TOKENIZER_DEV_ID
#define TOKENIZER_DEV_ID 81
#endif

#ifndef SAMPLER_DEV_ID
#define SAMPLER_DEV_ID 82
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

#define ASSERT(condition) do {if (!(condition)) { \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " Assertion failed: " + std::string(#condition)); \
}} while(0)