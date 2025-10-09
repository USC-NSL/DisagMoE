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
inline std::optional<std::vector<T>> slice_vector(const std::optional<std::vector<T>> &a, int l, int r) {
    if (!a.has_value()) return std::nullopt;
    return slice_vector(*a, l, r);
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

template<class T>
inline std::optional<std::vector<T>> duplicate_vector(const std::optional<std::vector<T>> &a, int times) {
    if (!a.has_value()) return std::nullopt;
    return duplicate_vector(*a, times);
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
inline std::optional<std::vector<T>> permute_vector(const std::optional<std::vector<T>> &data, const std::vector<int> &positions) {
    if (!data.has_value()) return std::nullopt;
    return permute_vector(*data, positions);
}

#endif