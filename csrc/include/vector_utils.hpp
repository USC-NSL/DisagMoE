#pragma once

#ifndef VECTOR_UTILS_H_
#define VECTOR_UTILS_H_

#include <vector>
#include <cassert>
#include <utility>

template<class T>
inline std::vector<T> slice_vector(const std::vector<T> &a, int l, int r) {
    if (a.empty() || l == r) return {};
    ASSERT(l <= r && l >= 0 && r <= a.size());
    return std::vector<T>(a.begin() + l, a.begin() + r);
}

template<class T>
inline std::vector<T> duplicate_vector(const std::vector<T> &a, int times) {
    std::vector<T> res;
    res.reserve(a.size() * times);
    for (int i = 0; i < times; i++) {
        res.insert(res.end(), a.begin(), a.end());
    }
    return res;
}

template<typename T>
std::vector<T> permute_vector(const std::vector<T> &data, const std::vector<int> &positions) {
    if (data.empty()) return {};
    std::vector<T> result(data.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        result[positions[i]] = data[i];
    }
    return result;
}

template<typename T>
std::vector<T> index_select_vector(const std::vector<T> &data, const std::vector<int> &indices) {
    if (data.empty()) return {};
    std::vector<T> result(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        result[i] = data[indices[i]];
    }
    return result;
}

template<class T>
std::vector<std::vector<T>> split_vector_by_indice(const std::vector<T> &vec, const std::vector<int> &indices) {
    // NOTE: will split to [indices[0], indices[1]), [indices[1], indices[2]), ..., [indices[n-1], indices[n])
    if (vec.empty()) {
        return std::vector<std::vector<T>>(indices.size() - 1, std::vector<T>());
    }
    std::vector<std::vector<T>> res{};
    for (size_t i = 0; i < indices.size() - 1; i ++) {
        int l = indices[i];
        int r = indices[i + 1];
        res.emplace_back(std::vector<T>(vec.begin() + l, vec.begin() + r));
    }
    return res;
}

template<class T>
std::vector<std::vector<T>> split_vector_by_size(const std::vector<T> &vec, const std::vector<int> &sizes) {
    if (vec.empty()) {
        return std::vector<std::vector<T>>(sizes.size(), std::vector<T>());
    }
    std::vector<std::vector<T>> res{};
    int base = 0;
    for (size_t i = 0; i < sizes.size(); i ++) {
        res.emplace_back(vec.begin() + base, vec.begin() + base + sizes[i]);
        base += sizes[i];
    }
    return res;
}

// NOTE: optional version

template<class T>
inline std::optional<std::vector<T>> slice_vector(const std::optional<std::vector<T>> &a, int l, int r) {
    if (!a.has_value()) return std::nullopt;
    return slice_vector(*a, l, r);
}

template<class T>
inline std::optional<std::vector<T>> duplicate_vector(const std::optional<std::vector<T>> &a, int times) {
    if (!a.has_value()) return std::nullopt;
    return duplicate_vector(*a, times);
}


template<typename T>
inline std::optional<std::vector<T>> permute_vector(const std::optional<std::vector<T>> &data, const std::vector<int> &positions) {
    if (!data.has_value()) return std::nullopt;
    return permute_vector(*data, positions);
}

template<typename T>
inline std::optional<std::vector<T>> index_select_vector(const std::optional<std::vector<T>> &data, const std::vector<int> &indices) {
    if (!data.has_value()) return std::nullopt;
    return index_select_vector(*data, indices);
}

template<class T>
inline std::optional<std::vector<std::vector<T>>> split_vector_by_indice(const std::optional<std::vector<T>> &vec, const std::vector<int> &indices) {
    if (!vec.has_value()) return std::nullopt;
    return split_vector_by_indice(*vec, indices);
}

template<class T>
inline std::optional<std::vector<std::vector<T>>> split_vector_by_size(const std::optional<std::vector<T>> &vec, const std::vector<int> &sizes) {
    if (!vec.has_value()) return std::nullopt;
    return split_vector_by_size(*vec, sizes);
}

#endif