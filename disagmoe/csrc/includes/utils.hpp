#pragma once

#include <cereal/archives/binary.hpp>
#include <sstream>
#include <vector>
#include <map>

#include "datatypes.hpp"
#include "cuda_utils.h"
#include "constants.h"
#include "logging.h"
#include "nccl.h"

inline uintptr_t tensor_at(uintptr_t buf, const Metadata& metadata, int i) {
    return buf + i * metadata.num_element() / metadata.num_tokens();
}

inline uintptr_t tensor_slice(uintptr_t src, const Metadata& metadata, const std::vector<int> &ids) {
    // TODO(hogura|20241001): rewrite this function as a cuda kernel for better performance
    int device;
    CUDACHECK(cudaGetDevice(&device));

    LOG(DEBUG) << "tensor_slice deivce: " << device << LEND;
    LOG(DEBUG) << "num_ele " << metadata.num_element() << " num_tokens " << metadata.num_tokens() << LEND;

    size_t count_per_token = metadata.num_element() / metadata.num_tokens();
    size_t size_per_token = count_per_token * metadata.get_datatype_size();
    uintptr_t dst = alloc_cuda_tensor(ids.size() * count_per_token, device);

    for (size_t i = 0; i < ids.size(); i ++) {
        int id = ids[i];
        // TODO(hogura|20241001): replace cudaMemcpy to cudaMemcpyAsync
        CUDACHECK(cudaMemcpy(
            (void*) (dst + i * size_per_token), 
            (void*) (src + id * size_per_token), 
            /*size=*/ size_per_token, 
            cudaMemcpyKind::cudaMemcpyDeviceToDevice
        ));
    }

    LOG(DEBUG) << "copied " << ids.size() << " tokens." << LEND;
    return dst;
}

template<class T, class T_COMP = std::less<T>>
inline std::vector<std::pair<T, TensorBatch>> group_by(
    uintptr_t buf, 
    const Metadata &metadata,
    const std::vector<T> &keys) {

    std::map<T, std::vector<int>, T_COMP> ids;
    ids.clear();

    LOG(DEBUG) << "gather #keys=" << keys.size() << LEND;

    for (size_t i = 0; i < keys.size(); i ++) {
        auto iter = ids.find(keys[i]);
        if (iter == ids.end()) {
            ids[keys[i]] = {i};
        } else {
            iter->second.push_back(i);
        }
    }

    std::vector<std::pair<T, TensorBatch>> results;
    for (auto &[key, grouped_ids]: ids) {
        LOG(DEBUG) << grouped_ids.size() << " #ids" << LEND;
        auto sliced_meta = metadata.at(grouped_ids); 
        auto sliced_tensor = tensor_slice(buf, metadata, grouped_ids);
        results.push_back(std::make_pair(
            key, 
            (TensorBatch) {
                sliced_tensor, 
                std::make_shared<Metadata>(sliced_meta)
            }
        ));
    }

    return results;
}

inline void* convert_to_nccl_uid(char* bytes) {
    // FIXME(hogura|20241003): the buf here never recycled actually
    size_t n = sizeof(ncclUniqueId::internal);

    char* buf = (char*) std::malloc(n);
    memcpy(buf, bytes, n);
    return (void*) buf;
}


inline std::string get_zmq_addr(int device_id) {
    char ip[256];
    sprintf(ip, "tcp://127.0.0.1:%d\0", ZMQ_PORT_BASE + device_id);
    return std::string(ip);
}


inline bool is_embedding_node(int device_id) {
    return device_id / 8 >= 100;
}


std::string cerealize(metadata_t metadata) {
    // use cereal to serialize metadata
    std::stringstream ss;
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(*metadata);
    return ss.str();
}

metadata_t decerealize(char* buf, size_t n) {
    std::string buffer(buf, n);
    std::istringstream ss(buffer);
    cereal::BinaryInputArchive iarchive(ss);
    Metadata result;
    iarchive(result);
    return std::make_shared<Metadata>(result);
}