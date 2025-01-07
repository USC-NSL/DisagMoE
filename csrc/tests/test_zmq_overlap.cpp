#include "tests.h"
#include "comm.h"

using Tensor = torch::Tensor;

void test_zmq_overlap(int rank) {
    const int bs = 256;
    const int hs = 4096;

    Metadata meta = Metadata {
        {bs, hs},
        "bf16",
        0,
        std::vector<int>(bs, 0),
        std::vector<int>(bs, 0),
        std::vector<int>(bs, 0),
        std::vector<int>(bs, 0)
    };
    if (rank == 82) {
        Tensor a = torch::empty(
            {bs, hs}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU)
        );
        auto c_send = create_zmq_channel(rank, 0, true);
        auto c_recv = create_zmq_channel(rank, 1, false);
        c_send->instantiate();
        c_recv->instantiate();
        c_recv->recv(a, meta);
        DMOE_LOG(INFO) << "Sampler Received from <1>: " << a.mean().item<float>() << LEND;
        c_send->send(a, meta);
        DMOE_LOG(INFO) << "Sampler Sent to <0>" << LEND;
    } else {
        Tensor a = torch::ones(
            {bs, hs}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0)
        ) * (rank + 1);
        if (rank == 0) {
            // receive from zmq
            auto c = create_zmq_channel(0, SAMPLER_DEV_ID, false);
            c->instantiate();
            c->recv(a, meta);
            DMOE_LOG(INFO) << "<0> Received from sampler: " << a.mean().item<float>() << LEND;
        } else if (rank == 1) {
            // send to zmq
            auto c = create_zmq_channel(1, SAMPLER_DEV_ID, true);
            c->instantiate();
            c->send(a, meta);
            DMOE_LOG(INFO) << "<1> Sent to sampler: " << a.mean().item<float>() << LEND;
        }
    }
}