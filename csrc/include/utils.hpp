#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>

#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
#include "cuda_utils.h"
#include "constants.h"
#include "logging.h"
#include "nccl.h"

inline clock_t t_now() {
    // Monotonic wall-clock time in microseconds
    return (clock_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

inline long long t_now_high() {
    auto now = std::chrono::system_clock::now();
    return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

inline torch::Tensor torch_tensor_slice(torch::Tensor tensor, const std::vector<int> &ids) {
    return tensor.index({
        torch::tensor(ids, torch::TensorOptions().dtype(torch::kInt32).device(tensor.device()))
    });
}

inline at::cuda::CUDAStream get_new_torch_stream() {
    at::cuda::CUDAStream c10_stream = at::cuda::getStreamFromPool(true, -1);
    return c10_stream;
}

inline uintptr_t tensor_at(uintptr_t buf, const BatchMetadata& metadata, int i) {
    return buf + i * metadata.num_element() / metadata.num_tokens() * metadata.get_datatype_size();
}

inline uintptr_t tensor_at(uintptr_t buf, batch_metadata_t metadata, int i) {
    return tensor_at(buf, *metadata, i);
}

#ifndef NCCL_UNIQUE_ID_BYTES
#define NCCL_UNIQUE_ID_BYTES 128
#endif

inline ncclUniqueId string_to_nccl_unique_id(const std::string& s) {
    // Validate the size (NCCL unique ID is always 128 bytes)
    if (s.size() != NCCL_UNIQUE_ID_BYTES) {
        throw std::runtime_error(
            "Invalid NCCL unique ID size: expected 128 bytes, got " + std::to_string(s.size()));
    }

    ncclUniqueId id;
    std::memcpy(id.internal, s.data(), NCCL_UNIQUE_ID_BYTES);
    return id;
}

template<class type>
std::string static cerealize(std::shared_ptr<type> metadata) {
    // use cereal to serialize metadata
    std::stringstream ss;
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(*metadata);
    return ss.str();
}

template<class type>
inline std::string static cerealize_(type data) {
    std::stringstream ss;
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(data);
    return ss.str();
}

template<class type>
std::shared_ptr<type> static decerealize(char* buf, size_t n) {
    std::string buffer(buf, n);
    std::istringstream ss(buffer);
    cereal::BinaryInputArchive iarchive(ss);
    type result;
    iarchive(result);
    // DMOE_LOG(WARNING) << "after decerealize, got metadata: " << result << LEND;
    return std::make_shared<type>(std::move(result));
}

template<class type>
inline void static decerealize_(char* buf, size_t n, type& result) {
    std::string buffer(buf, n);
    std::istringstream ss(buffer);
    cereal::BinaryInputArchive iarchive(ss);
    iarchive(result);
}

static void print_buf(void* buf, size_t n) {
    std::cerr << std::showbase << std::internal << std::setfill('0');
    uint8_t* data = (uint8_t*) buf;
    for (int i = 0; i < n; i ++)
        std::cerr << std::hex << std::setw(4) << data[i] << std::dec;
    std::cerr << std::endl;
}

template<class T> 
inline T range_max(const std::vector<T> &a) {
    T res;
    memset(&res, 0, sizeof(res));
    for (auto v: a)
        res = std::max(v, res);
    return res;
}