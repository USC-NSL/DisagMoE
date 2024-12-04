#include "cuda_utils.h"
#include "comm.h"
#include "datatypes.hpp"
#include "tests.h"
#include "permute.h"

#include <thread>

std::thread t_calc_op, t_comm_op, t_copy_op, t_kernel_op;

class Barrier {
    int n;
    int count = 0;
    std::mutex mtx;
    std::condition_variable cv;

public:
    Barrier(int n = 2): n(n) {

    }

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        count ++;
        if (count == n) {
            count = 0;
            cv.notify_all();
        } else {
            cv.wait(lock);
        }
    }

};

Barrier barrier;

inline float now() {
    return 1.0 * clock() / CLOCKS_PER_SEC;
}

void test_comm(int rank, std::vector<int> ranks, std::string uid) {
    // auto c_raw = create_nccl_group_channel(rank, ranks, (void*) uid.c_str());
    // c_raw->instantiate();
    // auto c = std::dynamic_pointer_cast<NcclGroupChannel>(c_raw);
    cudaSetDevice(0);
    ncclComm_t comm;
    ncclUniqueId& id = *((ncclUniqueId*)(uid.c_str()));
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    DMOE_LOG(INFO) << "rank " << rank << " ncclCommInitRankConfig." << now() << LEND;
    auto res = ncclCommInitRankConfig(&comm, ranks.size(), id, rank, &config);
    DMOE_LOG(INFO) << "rank " << rank << " initing..." << res << " " << now() << LEND;
    ncclResult_t state;
    do {
        ncclCommGetAsyncError(comm, &state);
    } while(state == ncclInProgress);
    at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAStreamGuard guard(c10_stream);
    auto cstream = c10_stream.stream();

    size_t size = 4;
    DMOE_LOG(WARNING) << "comm stream: " << cstream << LEND;
    torch::Tensor data = torch::empty({1, size}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
    uintptr_t buf = (uintptr_t) data.data_ptr();
    Metadata meta = Metadata {
        /*shape=*/ std::vector<size_t>({1, size}),
        /*dtype=*/ "fp16",
        /*layer_id=*/ 0,
        /*req_ids=*/ std::vector<int>({0}),
        /*exp_ids=*/ std::vector<int>({3}),
        /*prefill_poss=*/ std::vector<int>({4}),
    };
    CUDACHECK(cudaStreamSynchronize(cstream));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));

    barrier.arrive_and_wait();
    DMOE_LOG(INFO) << "rank " << rank << " communication start." << now() << LEND;

    if (rank == 1) {
        // ncclRecv((void*) buf, size, ncclFloat16, 0, comm, stream);
        NCCLCHECK(ncclBroadcast((void*) buf, (void*) buf, size, ncclFloat16, ranks[0], comm, stream));
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        NCCLCHECK(ncclBroadcast((void*) buf, (void*) buf, size, ncclFloat16, ranks[0], comm, stream));
    }

    DMOE_LOG(INFO) << "rank " << rank << " nccl submitted. " << now() << LEND;

    CUDACHECK(cudaStreamSynchronize(stream));
    
    DMOE_LOG(INFO) << "rank " << rank << " communication done. " << now() << LEND;
}

void test_calc(int rank) {
    DMOE_LOG(INFO) << "rank " << rank << " calculation start." << now() << LEND;
    at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(true, 0);
    at::cuda::CUDAStreamGuard guard(c10_stream);
    auto stream = c10_stream.stream();
    DMOE_LOG(WARNING) << "calc stream: " << stream << LEND;
    size_t size = 4096;
    torch::Tensor a = torch::empty({1, size}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
    torch::Tensor b = torch::empty({size, size}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
    DMOE_LOG(INFO) << "fill a" << LEND;
    a.fill_(1.0);
    DMOE_LOG(INFO) << "fill b" << LEND;
    b.fill_(2.0);

    DMOE_LOG(INFO) << "calc c" << LEND;
    barrier.arrive_and_wait();
    torch::Tensor c = a * b;

    DMOE_LOG(INFO) << "calc gpu->cpu" << LEND;
    auto result = c.sum().item<float>();
    
    DMOE_LOG(INFO) << "rank " << rank << " calculation done with result: " << result << "; " << now() << LEND;
}

void test_copy(int rank) {
    if (rank == 0)
        return;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    DMOE_LOG(INFO) << "rank " << rank << " copy start." << LEND;

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));
    size_t size = 4096;
    void* buf_cpu = std::malloc(size);
    void* buf_gpu;
    CUDACHECK(cudaMalloc(&buf_gpu, size));
    CUDACHECK(cudaMemcpyAsync(buf_gpu, buf_cpu, size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaMemcpyAsync(buf_cpu, buf_gpu, size, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    DMOE_LOG(INFO) << "rank " << rank << " copy done." << LEND;
}

void test_kernel_simple(int rank) {
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));
    size_t size = 4096;
    float* buf;
    CUDACHECK(cudaMalloc(&buf, size));
    add_one_cuda(buf, buf, 128, stream);
    add_one_cuda(buf, buf, 128, stream);
    add_one_cuda(buf, buf, 128, stream);
    cudaStreamSynchronize(stream);
    barrier.arrive_and_wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    DMOE_LOG(INFO) << "rank " << rank << " simple kernel start. " << now() << LEND;
    add_one_cuda(buf, buf, 128, stream);
    DMOE_LOG(INFO) << "rank " << rank << " simple kernel submitted. " << now() << LEND;
    cudaStreamSynchronize(stream);
    DMOE_LOG(INFO) << "rank " << rank << " simple kernel done." << now() << LEND;
}

void test_kernel(int rank) {
    if (rank == 0)
        return;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    DMOE_LOG(INFO) << "rank " << rank << " kernel start." << now() << LEND;
    at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(true, 0);
    at::cuda::CUDAStreamGuard guard(c10_stream);
    auto stream = c10_stream.stream();
    size_t size = 4096;
    Metadata meta = Metadata {
        /*shape=*/ std::vector<size_t>({4, size}),
        /*dtype=*/ "fp16",
        /*layer_id=*/ 0,
        /*req_ids=*/ std::vector<int>({0}),
        /*exp_ids=*/ std::vector<int>({3}),
        /*prefill_poss=*/ std::vector<int>({4}),
    };    
    torch::Tensor tensor = torch::empty({4, size}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));

    std::vector<uintptr_t> src_ptrs(meta.num_tokens());
    for (int i = 0; i < meta.num_tokens(); i ++) {
        void* buf;
        CUDACHECK(cudaMalloc(&buf, size * 4));
        src_ptrs[i] = (uintptr_t) buf;
    }

    DMOE_LOG(WARNING) << "kernel stream: " << stream << LEND;
    gather_tokens_cuda(tensor, src_ptrs.data(), meta.num_tokens(), meta.token_hidden_dim(), stream);
    cudaStreamSynchronize(stream);
    DMOE_LOG(INFO) << "rank " << rank << " kernel done." << LEND;
}

void test_op_overlap(int rank, std::vector<int> ranks, std::string uid) {
    t_comm_op = std::thread(test_comm, rank, ranks, uid);
    // t_calc_op = std::thread(test_calc, rank);
    // t_copy_op = std::thread(test_copy, rank);
    t_kernel_op = std::thread(test_kernel_simple, rank);
    t_comm_op.join();
    // t_calc_op.join();
    // t_copy_op.join();
    t_kernel_op.join();
}