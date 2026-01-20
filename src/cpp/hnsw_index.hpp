#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <queue>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "distance.hpp"

namespace vecdb {

class DimensionError : public std::runtime_error {
public:
    explicit DimensionError(const std::string& msg) : std::runtime_error(msg) {}
};

class DuplicateIDError : public std::runtime_error {
public:
    explicit DuplicateIDError(const std::string& msg) : std::runtime_error(msg) {}
};

class DeserializationError : public std::runtime_error {
public:
    explicit DeserializationError(const std::string& msg) : std::runtime_error(msg) {}
};

class HNSWIndex {
public:
    HNSWIndex(size_t dimension, const std::string& metric, size_t M = 16, size_t ef_construction = 200)
        : dimension_(dimension), metric_(metric), M_(M), M_max0_(M * 2),
          ef_construction_(ef_construction), entry_point_(-1), max_level_(0),
          dist_func_(get_distance_func(metric)), rng_(42), level_mult_(1.0 / std::log(static_cast<double>(M))) {
        if (dimension < 1) throw std::invalid_argument("Dimension must be >= 1");
    }

    void add(int64_t id, const std::vector<float>& vec) {
        if (vec.size() != dimension_)
            throw DimensionError("Expected " + std::to_string(dimension_) + ", got " + std::to_string(vec.size()));
        if (vectors_.count(id) && !deleted_.count(id))
            throw DuplicateIDError("ID " + std::to_string(id) + " already exists");

        vectors_[id] = vec;
        deleted_.erase(id);
        int level = random_level();
        levels_[id] = level;
        neighbors_[id].assign(level + 1, std::vector<int64_t>());

        if (entry_point_ < 0) {
            entry_point_ = id;
            max_level_ = level;
            return;
        }

        int64_t curr = entry_point_;
        // Traverse from top to insertion level
        for (int l = max_level_; l > level; --l) {
            curr = search_layer_single(vec.data(), curr, l);
        }

        // Insert at each level from min(level, max_level_) down to 0
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto candidates = search_layer(vec.data(), curr, ef_construction_, l);
            auto neighbors = select_neighbors(candidates, l == 0 ? M_max0_ : M_);
            neighbors_[id][l] = neighbors;

            // Bidirectional connections
            for (int64_t n : neighbors) {
                neighbors_[n][l].push_back(id);
                size_t max_conn = (l == 0 ? M_max0_ : M_);
                if (neighbors_[n][l].size() > max_conn) {
                    auto pruned = select_neighbors(
                        get_neighbors_with_dist(vectors_[n].data(), neighbors_[n][l]), max_conn);
                    neighbors_[n][l] = pruned;
                }
            }
            if (!candidates.empty()) curr = candidates[0].second;
        }

        if (level > max_level_) {
            max_level_ = level;
            entry_point_ = id;
        }
    }

    std::vector<std::pair<int64_t, float>> search(const std::vector<float>& query, size_t k, size_t ef_search = 50) {
        if (query.size() != dimension_)
            throw DimensionError("Expected " + std::to_string(dimension_) + ", got " + std::to_string(query.size()));
        if (k < 1) throw std::invalid_argument("k must be >= 1");
        if (entry_point_ < 0 || count() == 0) return {};

        int64_t curr = entry_point_;
        for (int l = max_level_; l > 0; --l) {
            curr = search_layer_single(query.data(), curr, l);
        }

        auto candidates = search_layer(query.data(), curr, std::max(ef_search, k), 0);

        std::vector<std::pair<int64_t, float>> results;
        for (auto& [dist, id] : candidates) {
            if (!deleted_.count(id)) {
                results.emplace_back(id, dist);
                if (results.size() >= k) break;
            }
        }
        return results;
    }

    bool remove(int64_t id) {
        if (!vectors_.count(id) || deleted_.count(id)) return false;
        deleted_.insert(id);
        return true;
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> data;
        auto write = [&data](const void* ptr, size_t size) {
            const uint8_t* p = static_cast<const uint8_t*>(ptr);
            data.insert(data.end(), p, p + size);
        };
        uint32_t version = 1;
        write(&version, 4);
        write(&dimension_, 8);
        uint32_t metric_len = static_cast<uint32_t>(metric_.size());
        write(&metric_len, 4);
        write(metric_.data(), metric_len);
        write(&M_, 8); write(&ef_construction_, 8);
        write(&entry_point_, 8); write(&max_level_, 4);

        uint64_t num_vectors = vectors_.size();
        write(&num_vectors, 8);
        for (const auto& [id, vec] : vectors_) {
            write(&id, 8);
            int level = levels_.at(id);
            write(&level, 4);
            write(vec.data(), dimension_ * 4);
            uint8_t is_del = deleted_.count(id) ? 1 : 0;
            write(&is_del, 1);
            for (int l = 0; l <= level; ++l) {
                uint32_t nn = static_cast<uint32_t>(neighbors_.at(id)[l].size());
                write(&nn, 4);
                write(neighbors_.at(id)[l].data(), nn * 8);
            }
        }
        return data;
    }

    static HNSWIndex deserialize(const std::vector<uint8_t>& data) {
        size_t offset = 0;
        auto read = [&data, &offset](void* ptr, size_t size) {
            if (offset + size > data.size()) throw DeserializationError("Unexpected end of data");
            std::memcpy(ptr, data.data() + offset, size);
            offset += size;
        };
        uint32_t version; read(&version, 4);
        if (version != 1) throw DeserializationError("Unsupported version");

        size_t dim; read(&dim, 8);
        uint32_t metric_len; read(&metric_len, 4);
        std::string metric(metric_len, '\0');
        read(metric.data(), metric_len);
        size_t M, ef_c; read(&M, 8); read(&ef_c, 8);

        HNSWIndex index(dim, metric, M, ef_c);
        read(&index.entry_point_, 8); read(&index.max_level_, 4);

        uint64_t num_vectors; read(&num_vectors, 8);
        for (uint64_t i = 0; i < num_vectors; ++i) {
            int64_t id; read(&id, 8);
            int level; read(&level, 4);
            std::vector<float> vec(dim);
            read(vec.data(), dim * 4);
            uint8_t is_del; read(&is_del, 1);
            index.vectors_[id] = vec;
            index.levels_[id] = level;
            if (is_del) index.deleted_.insert(id);
            index.neighbors_[id].resize(level + 1);
            for (int l = 0; l <= level; ++l) {
                uint32_t nn; read(&nn, 4);
                index.neighbors_[id][l].resize(nn);
                read(index.neighbors_[id][l].data(), nn * 8);
            }
        }
        return index;
    }

    size_t count() const { return vectors_.size() - deleted_.size(); }
    size_t dimension() const { return dimension_; }
    const std::string& metric() const { return metric_; }
    size_t M() const { return M_; }
    size_t ef_construction() const { return ef_construction_; }

private:
    using DistIDPair = std::pair<float, int64_t>;

    int random_level() {
        std::uniform_real_distribution<> dist(0.0, 1.0);
        int level = static_cast<int>(-std::log(dist(rng_)) * level_mult_);
        return std::max(0, level);
    }

    int64_t search_layer_single(const float* q, int64_t ep, int level) {
        float best_dist = dist_func_(q, vectors_[ep].data(), dimension_);
        int64_t best = ep;
        bool changed = true;
        while (changed) {
            changed = false;
            for (int64_t n : neighbors_[best][level]) {
                float d = dist_func_(q, vectors_[n].data(), dimension_);
                if (d < best_dist) {
                    best_dist = d;
                    best = n;
                    changed = true;
                }
            }
        }
        return best;
    }

    std::vector<DistIDPair> search_layer(const float* q, int64_t ep, size_t ef, int level) {
        std::unordered_set<int64_t> visited;
        // Min-heap for candidates to explore
        std::priority_queue<DistIDPair, std::vector<DistIDPair>, std::greater<DistIDPair>> candidates;
        // Max-heap for results (to easily remove worst)
        std::priority_queue<DistIDPair> results;

        float d = dist_func_(q, vectors_[ep].data(), dimension_);
        candidates.emplace(d, ep);
        results.emplace(d, ep);
        visited.insert(ep);

        while (!candidates.empty()) {
            auto [cd, cid] = candidates.top();
            candidates.pop();

            if (cd > results.top().first) break;

            for (int64_t n : neighbors_[cid][level]) {
                if (visited.insert(n).second) {
                    float nd = dist_func_(q, vectors_[n].data(), dimension_);
                    if (results.size() < ef || nd < results.top().first) {
                        candidates.emplace(nd, n);
                        results.emplace(nd, n);
                        if (results.size() > ef) results.pop();
                    }
                }
            }
        }

        std::vector<DistIDPair> result;
        while (!results.empty()) {
            result.push_back(results.top());
            results.pop();
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    std::vector<DistIDPair> get_neighbors_with_dist(const float* q, const std::vector<int64_t>& ids) {
        std::vector<DistIDPair> result;
        for (int64_t id : ids) {
            float d = dist_func_(q, vectors_[id].data(), dimension_);
            result.emplace_back(d, id);
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    std::vector<int64_t> select_neighbors(const std::vector<DistIDPair>& candidates, size_t M_cur) {
        std::vector<int64_t> result;
        for (const auto& [d, id] : candidates) {
            if (result.size() >= M_cur) break;
            result.push_back(id);
        }
        return result;
    }

    size_t dimension_;
    std::string metric_;
    size_t M_, M_max0_, ef_construction_;
    int64_t entry_point_;
    int max_level_;
    DistanceFunc dist_func_;
    std::mt19937 rng_;
    double level_mult_;
    std::unordered_map<int64_t, std::vector<float>> vectors_;
    std::unordered_map<int64_t, int> levels_;
    std::unordered_map<int64_t, std::vector<std::vector<int64_t>>> neighbors_;
    std::unordered_set<int64_t> deleted_;
};

} // namespace vecdb
