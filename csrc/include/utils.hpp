#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>

#include "datatypes.hpp"
#include "cuda_utils.h"
#include "constants.h"
#include "logging.h"
#include "nccl.h"

#define t_now clock

inline clock_t t_now_high() {
    auto now = std::chrono::system_clock::now();
    return (clock_t) std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

inline torch::Tensor torch_tensor_slice(torch::Tensor tensor, const std::vector<int> &ids) {
    return tensor.index({
        torch::tensor(ids, torch::TensorOptions().dtype(torch::kInt32).device(tensor.device()))
    });
}

inline uintptr_t tensor_at(uintptr_t buf, const Metadata& metadata, int i) {
    return buf + i * metadata.num_element() / metadata.num_tokens() * metadata.get_datatype_size();
}

inline uintptr_t tensor_at(uintptr_t buf, metadata_t metadata, int i) {
    return tensor_at(buf, *metadata, i);
}

inline uintptr_t tensor_slice(uintptr_t src, 
                              const Metadata& metadata, 
                              const std::vector<int> &ids,
                              bool on_gpu = true) {
    // TODO(hogura|20241001): rewrite this function as a cuda kernel for better performance
    // DMOE_LOG(DEBUG) << "num_ele " << metadata.num_element() << " num_tokens " << metadata.num_tokens() << LEND;

    size_t count_per_token = metadata.num_element() / metadata.num_tokens();
    size_t size_per_token = count_per_token * metadata.get_datatype_size();

    uintptr_t dst;
    if (on_gpu) {
        int device;
        CUDACHECK(cudaGetDevice(&device));
        // DMOE_LOG(DEBUG) << "tensor_slice deivce: " << device << LEND;
        dst = alloc_cuda_tensor(ids.size() * count_per_token, device);

        for (size_t i = 0; i < ids.size(); i ++) {
            int id = ids[i];
            // TODO(hogura|20241001): replace cudaMemcpy to cudaMemcpyAsync
            // !FIXME(hogura|20241010): the selection method here may have row-/col- wise issue.
            CUDACHECK(cudaMemcpy(
                (void*) (dst + i * size_per_token), 
                (void*) (src + id * size_per_token), 
                /*size=*/ size_per_token, 
                cudaMemcpyKind::cudaMemcpyDeviceToDevice
            ));
        }
    } else {
        void* buf = std::malloc(ids.size() * size_per_token);
        // TODO(hogura|20241007): use `omp parallel for` to accelerate
        for (int i = 0; i < ids.size(); i ++) {
            int id = ids[i];
            memcpy(
                buf + i * size_per_token,
                (void*) (src + id * size_per_token),
                size_per_token
            );
        }
        dst = (uintptr_t) buf;
    }

    // DMOE_LOG(DEBUG) << "copied " << ids.size() << " tokens." << LEND;
    return dst;
}

inline uintptr_t tensor_slice(uintptr_t src, 
                              metadata_t metadata, 
                              const std::vector<int> &ids,
                              bool on_gpu = true) {
    return tensor_slice(src, *metadata, ids, on_gpu);
}

template<class T, class T_COMP = std::less<T>>
inline std::vector<std::tuple<T, torch::Tensor, Metadata>> group_by(
    torch::Tensor tensor, 
    const Metadata &metadata,
    const std::vector<T> &keys,
    bool on_gpu = true) {

    std::map<T, std::vector<int>, T_COMP> ids{};

    // DMOE_LOG(DEBUG) << "gather #keys=" << keys.size() << LEND;
    assert (keys.size() == metadata.req_ids.size());
    for (size_t i = 0; i < keys.size(); i ++) {
        auto iter = ids.find(keys[i]);
        if (iter == ids.end()) {
            ids[keys[i]] = {i};
        } else {
            iter->second.push_back(i);
        }
    }

    std::vector<std::tuple<T, torch::Tensor, Metadata>> results;
    results.clear();
    for (auto &[key, grouped_ids]: ids) {
        // DMOE_LOG(DEBUG) << grouped_ids.size() << " #ids" << LEND;
        Metadata sliced_meta = metadata.at(grouped_ids);
        auto sliced_tensor = torch_tensor_slice(tensor, grouped_ids);
        results.push_back(std::make_tuple(
            key, 
            sliced_tensor, 
            sliced_meta
        ));
    }

    // DMOE_LOG(DEBUG) << " Returning results" << LEND;

    return results;
}

inline void* convert_to_nccl_uid(char* bytes) {
    // FIXME(hogura|20241003): the buf here never recycled actually
    size_t n = sizeof(ncclUniqueId::internal);

    char* buf = (char*) std::malloc(n);
    memcpy(buf, bytes, n);
    return (void*) buf;
}

inline bool is_embedding_node(int device_id) {
    return device_id == TOKENIZER_DEV_ID || device_id == SAMPLER_DEV_ID;
}

inline bool is_tokenizer(int device_id) {
    return device_id == TOKENIZER_DEV_ID;
}

inline bool is_sampler(int device_id) {
    return device_id == SAMPLER_DEV_ID;
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
    Metadata result;
    iarchive(result);
    // DMOE_LOG(WARNING) << "after decerealize, got metadata: " << result << LEND;
    return std::make_shared<type>(result);
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