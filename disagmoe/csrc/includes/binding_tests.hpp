#pragma once

#include "comm.h"

#include <thread>

void test_nccl_p2p(Channel_t c1, uintptr_t p1, Channel_t c2, uintptr_t p2, const Metadata& metadata) {
    auto t1 = std::thread([&]{c1->send(p1, metadata);});
    auto t2 = std::thread([&]{c2->recv(p2, metadata);});
    t1.join();
    t2.join();
}