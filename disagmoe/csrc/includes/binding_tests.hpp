#pragma once

#include "comm.h"
#include "logging.h"

#include <thread>

void test_nccl_p2p(Channel_t c1, uintptr_t p1, Channel_t c2, uintptr_t p2, const Metadata& metadata) {
    auto t1 = std::thread([&]{c1->send(p1, metadata);});
    auto t2 = std::thread([&]{c2->recv(p2, metadata);});
    t1.join();
    t2.join();
}

std::pair<Channel_t, Channel_t> _init_channel(int s = 0, int r = 1) {
    auto uid = get_nccl_unique_id();
    auto c1 = create_channel(s, r, uid);
    auto c2 = create_channel(r, s, uid);

    auto t1 = std::thread([&]{ c1->instantiate(); }), t2 = std::thread([&]{ c2->instantiate(); });
    t1.join(); t2.join();
    return std::make_pair(c1, c2);
}

void test_zmq_sub_pub() {
    auto pr = _init_channel();
    auto c1 = pr.first, c2 = pr.second;

    MuAttnDispatcher sender({0}, 0, {c1});
    MuPool recver({0}, 1, {c2});

    sender.start();
    recver.start();

    int n = 1024;
    int m = 10;
    std::vector<TokenMetadata> infos(n);
    for (int i = 0; i < n; i ++)
        infos[i].req_id = i;

    auto meta = Metadata{
        {1024, 8192},
        "fp16",
        0,
        infos
    };
    printf("sizeof meta %u\n", sizeof(meta));
    auto batch = (TensorBatch){0, std::make_shared<Metadata>(meta)};
    for (int i = 0; i < m; i ++)
        sender.put(batch);

    for (;;);
}

void test_attn_dispatcher() {
    auto pr = _init_channel();
    auto c0 = pr.first, c1 = pr.second;

    MuAttnDispatcher sender({0, 1}, 0, {c0});
    MuPool recver({0, 1}, 1, {c1});

    sender.start();
    recver.start();

    puts("started sender & recver");

    int n = 10;
    int bs = 8;
    int hs = 8;

    std::vector<TokenMetadata> infos(n);
    for (int i = 0; i < n; i ++)
        infos[i].req_id = i;
    auto meta0 = Metadata{
        /*shape=*/ {bs, hs},
        "fp16",
        /*layer_id=*/ 0,
        infos
    };

    auto ptr0 = alloc_cuda_tensor(bs * hs, 0);

    auto meta1 = meta0; meta1.layer_id = 1;
    for (int i = 0; i < n; i ++) {
        meta1.infos[i].prefill_pos = 1;
        meta1.infos[i].first_attn_id = 233;
    }
    sender.put((TensorBatch) {ptr0, std::make_shared<Metadata>(meta0)});
    sender.put((TensorBatch) {ptr0, std::make_shared<Metadata>(meta1)});

    recver.wait_for_new_requests();
    puts("first fetch");
    auto res = recver.fetch_largest_batch();
    printf("fetched size: %u\n", res.size());
    assert(res.size() == 1);
    std::cout << *res[0].metadata << std::endl;

    recver.wait_for_new_requests();
    puts("second fetch");
    res = recver.fetch_largest_batch();
    printf("fetched size: %u\n", res.size());
    assert(res.size() == 1);
    std::cout << *res[0].metadata << std::endl;

    puts("passed");
    fflush(stdout);
    exit(0);
}

void test_expert_dispatcher() {
    auto pr = _init_channel(0, 2);
    auto c0 = pr.first, r1 = pr.second;
    pr = _init_channel(1, 2);
    auto c1 = pr.first, r2 = pr.second;

    MuExpertDispatcher sender0({0}, 0, {c0}, {ChannelInfo{{0}, {0}}});
    MuExpertDispatcher sender1({1}, 1, {c1}, {ChannelInfo{{0}, {1}}});
    MuPool recver({0, 1}, 2, {r1, r2});

    recver.start();
    sender0.start();
    sender1.start();

    puts("started sender0 & recver");

    int n = 8;
    size_t bs = 8;
    size_t hs = 8;

    std::vector<TokenMetadata> infos(n);
    for (int i = 0; i < n; i ++)
        infos[i].req_id = i;
    auto meta0 = Metadata{
        /*shape=*/ {bs, hs},
        "fp16",
        /*layer_id=*/ 0,
        infos
    };

    auto meta1 = meta0; meta1.layer_id = 1;
    for (int i = 0; i < n; i ++) {
        meta1.infos[i].prefill_pos = 1;
        meta1.infos[i].first_attn_id = 233;
    }
    for (int i = 0; i < 5; i ++) {
        auto n = (i + 1) * bs;
        meta0.shape = {n, hs};
        meta0.infos.resize(n);
        auto ptr0 = alloc_cuda_tensor(n * hs, 0);
        sender0.put((TensorBatch) {ptr0, std::make_shared<Metadata>(meta0)});
    }
    for (int i = 0; i < 5; i ++) {
        auto n = (i + 6) * bs;
        meta1.shape = {n, hs};
        meta1.infos.resize(n);
        auto ptr1 = alloc_cuda_tensor(n * hs, 1);
        sender1.put((TensorBatch) {ptr1, std::make_shared<Metadata>(meta1)});
    }

    sleep(1);

    recver.wait_for_new_requests();
    puts("first fetch");
    auto res = recver.fetch_largest_batch();
    printf("fetched size: %u\n", res.size());
    // if (res.size() == 2) {
    //     for (int i = 0; i < 2; i ++)
    //         LOG(INFO) << *res[i].metadata << LEND;
    //     LOG(INFO) << "passed" << LEND;
    //     exit(0);
    // }
    std::cout << *res[0].metadata << std::endl;

    recver.wait_for_new_requests();
    puts("second fetch");
    res = recver.fetch_largest_batch();
    printf("fetched size: %u\n", res.size());
    std::cout << *res[0].metadata << std::endl;

    puts("passed");
    fflush(stdout);
    exit(0);
}