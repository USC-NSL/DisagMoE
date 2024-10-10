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

inline uintptr_t tensor_at(uintptr_t buf, metadata_t metadata, int i) {
    return tensor_at(buf, *metadata, i);
}

inline uintptr_t tensor_slice(uintptr_t src, 
                              const Metadata& metadata, 
                              const std::vector<int> &ids,
                              bool on_gpu = true) {
    // TODO(hogura|20241001): rewrite this function as a cuda kernel for better performance
    // LOG(DEBUG) << "num_ele " << metadata.num_element() << " num_tokens " << metadata.num_tokens() << LEND;

    size_t count_per_token = metadata.num_element() / metadata.num_tokens();
    size_t size_per_token = count_per_token * metadata.get_datatype_size();

    uintptr_t dst;
    if (on_gpu) {
        int device;
        CUDACHECK(cudaGetDevice(&device));
        // LOG(DEBUG) << "tensor_slice deivce: " << device << LEND;
        dst = alloc_cuda_tensor(ids.size() * count_per_token, device);

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
    } else {
        void* buf = std::malloc(ids.size() * count_per_token);
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

    // LOG(DEBUG) << "copied " << ids.size() << " tokens." << LEND;
    return dst;
}

inline uintptr_t tensor_slice(uintptr_t src, 
                              metadata_t metadata, 
                              const std::vector<int> &ids,
                              bool on_gpu = true) {
    return tensor_slice(src, *metadata, ids, on_gpu);
}

template<class T, class T_COMP = std::less<T>>
inline std::vector<std::tuple<T, uintptr_t, Metadata>> group_by(
    uintptr_t buf, 
    const Metadata &metadata,
    const std::vector<T> &keys,
    bool on_gpu = true) {

    std::map<T, std::vector<int>, T_COMP> ids;
    ids.clear();

    // LOG(DEBUG) << "gather #keys=" << keys.size() << LEND;
    assert(keys.size() == metadata.infos.size());
    for (size_t i = 0; i < keys.size(); i ++) {
        auto iter = ids.find(keys[i]);
        if (iter == ids.end()) {
            ids[keys[i]] = {i};
        } else {
            iter->second.push_back(i);
        }
    }

    std::vector<std::tuple<T, uintptr_t, Metadata>> results;
    for (auto &[key, grouped_ids]: ids) {
        // LOG(DEBUG) << grouped_ids.size() << " #ids" << LEND;
        Metadata sliced_meta = metadata.at(grouped_ids); 
        auto sliced_tensor = tensor_slice(buf, metadata, grouped_ids, on_gpu);
        results.push_back(std::make_tuple(
            key, 
            sliced_tensor, 
            sliced_meta
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


inline std::string get_zmq_addr(int device_id, bool is_gpu = true) {
    char ip[256];
    sprintf(ip, "tcp://127.0.0.1:%d\0", (is_gpu ? ZMQ_PORT_BASE : ZMQ_CPU_PORT_BASE) + device_id);
    return std::string(ip);
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


std::string static cerealize(metadata_t metadata) {
    // use cereal to serialize metadata
    std::stringstream ss;
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(*metadata);
    return ss.str();
}

metadata_t static decerealize(char* buf, size_t n) {
    std::string buffer(buf, n);
    std::istringstream ss(buffer);
    cereal::BinaryInputArchive iarchive(ss);
    Metadata result;
    iarchive(result);
    LOG(WARNING) << "after decerealize, got metadata: " << result << LEND;
    return std::make_shared<Metadata>(result);
}

static void print_buf(void* buf, size_t n) {
    std::cerr << std::showbase << std::internal << std::setfill('0');
    uint8_t* data = (uint8_t*) buf;
    for (int i = 0; i < n; i ++)
        std::cerr << std::hex << std::setw(4) << data[i] << std::dec;
    std::cerr << std::endl;
}