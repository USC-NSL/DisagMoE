#pragma once

const int ZMQ_PORT_BASE = 24000;
const int ZMQ_CPU_PORT_BASE = 25000;

#ifndef MAX_OUTPUT_LEN
#define MAX_OUTPUT_LEN 64
#endif

#ifndef EOS_TOKEN_ID
#define EOS_TOKEN_ID 2
#endif

#ifndef SAMPLER_DEV_ID
#define SAMPLER_DEV_ID 81
#endif

#ifndef TOKENIZER_DEV_ID
#define TOKENIZER_DEV_ID 82
#endif