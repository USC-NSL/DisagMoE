#pragma once

#include "datatypes.hpp"

inline uintptr_t tensor_at(uintptr_t buf, const Metadata& metadata, int i) {
    return buf + i * metadata.num_element() / metadata.num_tokens();
}