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

template<class T>
inline std::vector<std::vector<T>> split_vector(const std::vector<T> &vec, const std::vector<int> &indices) {
    std::vector<std::vector<T>> res{};
    if (vec.empty()) return {};

    for (size_t i = 0; i < indices.size() - 1; i ++) {
        int l = indices[i];
        int r = indices[i + 1];
        res.emplace_back(std::vector<T>(vec.begin() + l, vec.begin() + r));
    }
    return res;
}

template<class T>
inline std::optional<std::vector<std::vector<T>>> split_vector(const std::optional<std::vector<T>> &vec, const std::vector<int> &indices) {
    if (!vec.has_value()) return std::nullopt;
    return split_vector(*vec, indices);
}

inline std::vector<torch::Tensor> split_tensor(const torch::Tensor &tensor, const std::vector<int> &indices) {
    std::vector<torch::Tensor> res{};
    ASSERT (tensor.size(0) == indices.back());

    for (size_t i = 0; i < indices.size() - 1; i ++) {
        int l = indices[i];
        int r = indices[i + 1];
        res.emplace_back(tensor.slice(0, l, r));
    }
    return res;
}

#endif