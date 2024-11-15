#include "cuda_utils.h"
#include "comm.h"

#include <thread>

std::thread t_calc, t_comm;

void test_bcast_comm(int rank, std::vector<int> ranks, std::string uid) {
    auto c_raw = create_nccl_group_channel(rank, ranks, (void*) uid.c_str());
    c_raw->instantiate();
    auto c = std::dynamic_pointer_cast<NcclGroupChannel>(c_raw);

    size_t size = 4096;
    auto buf = (void*) alloc_cuda_tensor(4096, rank, 1);
    c->bcast_obj(buf, size);

    if (rank == 0)
        std::this_thread::sleep_for(std::chrono::seconds(2));
    
    c->bcast_obj(buf, size);

    LOG(INFO) << "rank " << rank << " communication done." << LEND;
}

void test_bcast_calc(int rank) {
    
}