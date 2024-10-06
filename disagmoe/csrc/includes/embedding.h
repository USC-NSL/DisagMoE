#pragma once

#include "muhelper.h"

class Sampler: public MuHelper {
protected:
    void run() override;

public:
    Sampler(int device_id, std::vector<Channel_t> channels);
};

class Tokenizer: public MuHelper {
protected:
    void run() override;

public:
    Tokenizer(int device_id, std::vector<Channel_t> channels);
};