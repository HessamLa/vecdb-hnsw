#pragma once

#include <cstddef>
#include <cmath>
#include <string>
#include <stdexcept>

namespace vecdb {

// L2 (Euclidean) distance
inline float l2_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Cosine distance: 1 - cosine_similarity
inline float cosine_distance(const float* a, const float* b, size_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
    float sim = dot / (norm_a * norm_b);
    sim = std::max(-1.0f, std::min(1.0f, sim));
    return 1.0f - sim;
}

// Dot distance: -dot_product (for MIPS)
inline float dot_distance(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return -dot;
}

// Distance function type
using DistanceFunc = float (*)(const float*, const float*, size_t);

inline DistanceFunc get_distance_func(const std::string& metric) {
    if (metric == "l2") return l2_distance;
    if (metric == "cosine") return cosine_distance;
    if (metric == "dot") return dot_distance;
    throw std::invalid_argument("Invalid metric: " + metric);
}

} // namespace vecdb
