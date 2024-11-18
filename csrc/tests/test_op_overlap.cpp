#include "cuda_utils.h"
#include "comm.h"
#include "datatypes.hpp"
#include "tests.h"
#include "permute.h"

#include <thread>

std::thread t_calc_op, t_comm_op, t_copy_op, t_kernel_op;

void test_comm(int rank, std::vector<int> ranks, std::string uid) {
    auto c_raw = create_nccl_group_channel(rank, ranks, (void*) uid.c_str());
    c_raw->instantiate();
    auto c = std::dynamic_pointer_cast<NcclGroupChannel>(c_raw);
    DMOE_LOG(INFO) << "rank " << rank << " communication start." << LEND;
    at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAStreamGuard guard(c10_stream);
    auto stream = c10_stream.stream();

    size_t size = 4096;
    DMOE_LOG(WARNING) << "comm stream: " << stream << LEND;
    torch::Tensor data = torch::empty({1, size}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
    uintptr_t buf = (uintptr_t) data.data_ptr();
    Metadata meta = Metadata {
        /*shape=*/ std::vector<size_t>({1, size}),
        /*dtype=*/ "fp16",
        /*layer_id=*/ 0,
        /*req_ids=*/ std::vector<int>({0}),
        /*exp_ids=*/ std::vector<int>({3}),
        /*prefill_poss=*/ std::vector<int>({4}),
        /*prompt_lens=*/ std::map<int, int>(),
    };

    if (rank == 1)
        c->send_recv(buf, meta);
    
    DMOE_LOG(INFO) << "rank " << rank << " communication done." << LEND;
}

void test_calc(int rank) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    DMOE_LOG(INFO) << "rank " << rank << " calculation start." << LEND;
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
    torch::Tensor c = a * b;

    DMOE_LOG(INFO) << "calc gpu->cpu" << LEND;
    auto result = c.sum().item<float>();
    
    DMOE_LOG(INFO) << "rank " << rank << " calculation done with result: " << result << LEND;
}

void test_copy(int rank) {
    if (rank == 0)
        return;
    std::this_thread::sleep_for(std::chrono::seconds(2));
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

void test_kernel(int rank) {
    if (rank == 0)
        return;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    DMOE_LOG(INFO) << "rank " << rank << " kernel start." << LEND;
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
        /*prompt_lens=*/ std::map<int, int>()
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
    DMOE_LOG(INFO) << "rank " << rank << " kernel done." << LEND;
}

void test_op_overlap(int rank, std::vector<int> ranks, std::string uid) {
    t_comm_op = std::thread(test_comm, rank, ranks, uid);
    t_calc_op = std::thread(test_calc, rank);
    t_copy_op = std::thread(test_copy, rank);
    t_kernel_op = std::thread(test_kernel, rank);
    t_comm_op.join();
    t_calc_op.join();
    t_copy_op.join();
    t_kernel_op.join();
}