#pragma once

#include <vector>
#include <map>

#include "datatypes.hpp"
#include "cuda_utils.h"
#include "logging.h"

inline uintptr_t tensor_at(uintptr_t buf, const Metadata& metadata, int i) {
    return buf + i * metadata.num_element() / metadata.num_tokens();
}

inline uintptr_t tensor_slice(uintptr_t src, const Metadata& metadata, const std::vector<int> &ids) {
    // TODO(hogura|20241001): rewrite this function as a cuda kernel for better performance
    int device;
    CUDACHECK(cudaGetDevice(&device));

    LOG(DEBUG) << "num_ele " << metadata.num_element() << " num_tokens " << metadata.num_tokens() << LEND;

    size_t count_per_token = metadata.num_element() / metadata.num_tokens();
    size_t size_per_token = count_per_token * metadata.get_datatype_size();
    uintptr_t dst = alloc_cuda_tensor(ids.size() * count_per_token, device);

    for (size_t i = 0; i < ids.size(); i ++) {
        int id = ids[i];
        CUDACHECK(cudaMemcpyAsync(
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