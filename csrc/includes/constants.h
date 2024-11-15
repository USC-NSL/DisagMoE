#pragma once

#include <string>
#include <exception>

const int ZMQ_PORT_BASE = 24000;
const int ZMQ_CPU_PORT_BASE = 25000;

#ifndef MAX_OUTPUT_LEN
#define MAX_OUTPUT_LEN 16
#endif

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

#define ASSERT(condition) do {if (!(condition)) { \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " Assertion failed: " + std::string(#condition)); \
}} while(0)