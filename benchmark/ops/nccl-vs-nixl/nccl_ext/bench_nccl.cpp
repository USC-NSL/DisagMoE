#include <torch/extension.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstring>

namespace py = pybind11;

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                           \
    if (r != ncclSuccess) {                         \
        throw std::runtime_error(                   \
            std::string("NCCL error: ") +           \
            ncclGetErrorString(r));                  \
    }                                               \
} while(0)

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                            \
    if (e != cudaSuccess) {                         \
        throw std::runtime_error(                   \
            std::string("CUDA error: ") +           \
            cudaGetErrorString(e));                  \
    }                                               \
} while(0)

class BenchNcclChannel {
    ncclComm_t comm_;
    ncclUniqueId uid_;
    cudaStream_t stream_;
    int local_rank_;
    int remote_rank_;
    bool initialized_ = false;

public:
    BenchNcclChannel(int local_rank, int remote_rank, py::bytes uid_bytes)
        : local_rank_(local_rank), remote_rank_(remote_rank) {
        std::string s = uid_bytes;
        std::memcpy(&uid_, s.data(), sizeof(ncclUniqueId));
        CUDACHECK(cudaStreamCreate(&stream_));
    }

    void initialize() {
        NCCLCHECK(ncclCommInitRank(&comm_, 2, uid_, local_rank_));
        initialized_ = true;
    }

    void send(torch::Tensor t) {
        NCCLCHECK(ncclSend(t.data_ptr(), t.numel(),
                           ncclBfloat16, remote_rank_, comm_, stream_));
    }

    void recv(torch::Tensor t) {
        NCCLCHECK(ncclRecv(t.data_ptr(), t.numel(),
                           ncclBfloat16, remote_rank_, comm_, stream_));
    }

    void sync() {
        CUDACHECK(cudaStreamSynchronize(stream_));
    }

    ~BenchNcclChannel() {
        if (initialized_) ncclCommDestroy(comm_);
        cudaStreamDestroy(stream_);
    }
};

py::bytes get_nccl_unique_id_bytes() {
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    return py::bytes(reinterpret_cast<char*>(&id), sizeof(ncclUniqueId));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BenchNcclChannel>(m, "NcclChannel")
        .def(py::init<int, int, py::bytes>(),
             py::arg("local_rank"), py::arg("remote_rank"), py::arg("uid_bytes"))
        .def("initialize", &BenchNcclChannel::initialize)
        .def("send", &BenchNcclChannel::send)
        .def("recv", &BenchNcclChannel::recv)
        .def("sync", &BenchNcclChannel::sync);

    m.def("get_nccl_unique_id_bytes", &get_nccl_unique_id_bytes);
}
