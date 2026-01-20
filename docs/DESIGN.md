# VecDB Design Document

> **Project:** Minimal Viable Vector Database
> **Version:** 1.0
> **Date:** January 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Requirements Mapping](#2-requirements-mapping)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Components](#4-core-components)
5. [HNSW Algorithm](#5-hnsw-algorithm)
6. [Persistence Strategy](#6-persistence-strategy)
7. [API Reference](#7-api-reference)
8. [Trade-offs & Future Improvements](#8-trade-offs--future-improvements)

---

## 1. Executive Summary

VecDB is a minimal vector database implementing the HNSW (Hierarchical Navigable Small World) algorithm for efficient approximate nearest neighbor search. The system achieves **O(log N)** search complexity compared to brute-force O(N), with search latencies under 3ms for 10,000 vectors.

### Key Achievements

| Metric | Result |
|--------|--------|
| Search Complexity | O(log N) |
| Search Latency (10K vectors) | ~2ms |
| Recall@10 (1K vectors) | 98% |
| Tests Passing | 125/125 |
| Distance Metrics | L2, Cosine, Dot Product |

---

## 2. Requirements Mapping

This section maps each requirement from `project-description.md` to its implementation.

### 2.1 Core Requirements

#### A. Data Management

| Requirement | Implementation | Location |
|-------------|----------------|----------|
| Store vectors | `Collection.insert()` stores vectors in memory | `src/python/vecdb/collection.py:100-125` |
| Retrieve vectors | `Collection.get()` retrieves by user ID | `src/python/vecdb/collection.py:169-180` |
| Delete vectors | `Collection.delete()` with lazy deletion | `src/python/vecdb/collection.py:146-166` |
| Multiple collections | `VecDB.create_collection()` manages named collections | `src/python/vecdb/vecdb.py:60-85` |
| ID mapping | `user_to_internal` / `internal_to_user` dicts | `src/python/vecdb/collection.py:70-71` |

**Key Classes:**
```
src/python/vecdb/
├── collection.py    → Collection class (vector management)
└── vecdb.py         → VecDB class (database management)
```

#### B. Querying & Search

| Requirement | Implementation | Location |
|-------------|----------------|----------|
| Similarity search | `Collection.search()` → `HNSWIndex.search()` | `src/python/vecdb/collection.py:127-144` |
| k-NN queries | Returns top-k results sorted by distance | `src/cpp/hnsw_index.hpp:88-108` |
| Distance metrics | L2, Cosine, Dot Product | `src/cpp/distance.hpp:10-45` |
| Sub-linear search | HNSW graph traversal O(log N) | `src/cpp/hnsw_index.hpp:204-258` |

**Key Functions:**
```cpp
// src/cpp/distance.hpp
float l2_distance(const float* a, const float* b, size_t dim);
float cosine_distance(const float* a, const float* b, size_t dim);
float dot_distance(const float* a, const float* b, size_t dim);

// src/cpp/hnsw_index.hpp
std::vector<std::pair<int64_t, float>> HNSWIndex::search(query, k, ef_search);
```

#### C. Persistence

| Requirement | Implementation | Location |
|-------------|----------------|----------|
| Save to disk | `PersistenceManager.save_collection()` | `src/python/vecdb/persistence.py:54-75` |
| Load from disk | `PersistenceManager.load_collection()` | `src/python/vecdb/persistence.py:77-158` |
| Atomic writes | Write to `.tmp` then rename | `src/python/vecdb/persistence.py:187-199` |
| Binary format | HNSW serialization + struct-packed vectors | `src/cpp/hnsw_index.hpp:117-147` |

**File Structure:**
```
db_path/
├── metadata.json           # Database metadata
└── collections/
    ├── {name}.meta        # Collection config (JSON)
    ├── {name}.hnsw        # HNSW index (binary)
    └── {name}.vectors     # Vectors + ID mappings (binary)
```

### 2.2 Scalability Challenge

| Requirement | Implementation | Location |
|-------------|----------------|----------|
| Avoid O(N) brute-force | HNSW algorithm | `src/cpp/hnsw_index.hpp` |
| Multi-layer graph | Hierarchical navigation | `src/cpp/hnsw_index.hpp:39-86` |
| Greedy search | Layer-by-layer descent | `src/cpp/hnsw_index.hpp:204-220` |
| Configurable parameters | M, ef_construction, ef_search | `src/cpp/hnsw_index.hpp:32-36` |

**Complexity Analysis:**
- **Insert:** O(log N) - traverse layers + connect neighbors
- **Search:** O(log N) - greedy descent through layers
- **Delete:** O(1) - lazy deletion (mark as deleted)

### 2.3 Deliverables Checklist

| Deliverable | Status | Location |
|-------------|--------|----------|
| Source Code | ✅ | `src/cpp/`, `src/python/vecdb/` |
| README.md | ✅ | `README.md` |
| Build instructions | ✅ | `README.md#installation` |
| API reference | ✅ | `README.md#api-reference` |
| Design Doc | ✅ | This document (`docs/DESIGN.md`) |
| Trade-offs section | ✅ | Section 8 below |
| Unit tests | ✅ | `tests/python/test_*.py` (125 tests) |
| Integration tests | ✅ | `tests/python/test_integration.py` |
| Benchmarks | ✅ | `benchmarks/`, `docs/BENCHMARKS.md` |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Application                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API Layer (Python)                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  VecDB (src/python/vecdb/vecdb.py)                              │   │
│  │  - create_collection(), get_collection(), delete_collection()   │   │
│  │  - save(), close(), context manager                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Collection     │    │    HNSW Index       │    │   Persistence       │
│  Manager        │◄──►│    (C++)            │◄──►│   Manager           │
│  (Python)       │    │                     │    │   (Python)          │
├─────────────────┤    ├─────────────────────┤    ├─────────────────────┤
│ collection.py   │    │ hnsw_index.hpp      │    │ persistence.py      │
│                 │    │ distance.hpp        │    │                     │
│ - insert()      │    │ bindings.cpp        │    │ - save_collection() │
│ - search()      │    │                     │    │ - load_collection() │
│ - get()         │    │ - add()             │    │ - delete_collection()│
│ - delete()      │    │ - search()          │    │ - list_collections()│
│ - ID mapping    │    │ - remove()          │    │ - atomic writes     │
│                 │    │ - serialize()       │    │                     │
└─────────────────┘    └─────────────────────┘    └─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Exception Classes                                │
│                    (src/python/vecdb/exceptions.py)                     │
│  VecDBError, DimensionError, DuplicateIDError, CollectionExistsError,  │
│  CollectionNotFoundError, DeserializationError                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Components

### 4.1 VecDB Class (`src/python/vecdb/vecdb.py`)

The main entry point for users. Manages collections and persistence.

```python
class VecDB:
    def __init__(self, path: str = './vecdb_data')
    def create_collection(self, name, dimension, metric='l2', hnsw_params=None) -> Collection
    def get_collection(self, name) -> Collection
    def delete_collection(self, name) -> bool
    def list_collections(self) -> List[str]
    def save(self) -> None
    def close(self) -> None
    def __enter__(self) / __exit__()  # Context manager
```

### 4.2 Collection Class (`src/python/vecdb/collection.py`)

Manages vectors within a collection, handles ID mapping.

```python
class Collection:
    def __init__(self, name, dimension, metric, hnsw_params=None)
    def insert(self, user_id: int, vector: List[float]) -> None
    def search(self, query: List[float], k=10, ef_search=50) -> List[Tuple[int, float]]
    def get(self, user_id: int) -> Optional[List[float]]
    def delete(self, user_id: int) -> bool
    def contains(self, user_id: int) -> bool
    def count(self) -> int
```

**Internal Data Structures:**
```python
_user_to_internal: Dict[int, int]  # User ID → HNSW internal ID
_internal_to_user: Dict[int, int]  # HNSW internal ID → User ID
_vectors: Dict[int, List[float]]   # User ID → Original vector
_hnsw_index: HNSWIndex             # C++ HNSW implementation
```

### 4.3 HNSWIndex Class (`src/cpp/hnsw_index.hpp`)

The core HNSW implementation in C++.

```cpp
class HNSWIndex {
public:
    HNSWIndex(size_t dimension, const std::string& metric,
              size_t M = 16, size_t ef_construction = 200);

    void add(int64_t id, const std::vector<float>& vec);
    std::vector<std::pair<int64_t, float>> search(
        const std::vector<float>& query, size_t k, size_t ef_search = 50);
    bool remove(int64_t id);

    std::vector<uint8_t> serialize() const;
    static HNSWIndex deserialize(const std::vector<uint8_t>& data);

    size_t count() const;
    size_t dimension() const;
    const std::string& metric() const;
};
```

### 4.4 Distance Functions (`src/cpp/distance.hpp`)

```cpp
namespace vecdb {
    float l2_distance(const float* a, const float* b, size_t dim);
    float cosine_distance(const float* a, const float* b, size_t dim);
    float dot_distance(const float* a, const float* b, size_t dim);
    DistanceFunc get_distance_func(const std::string& metric);
}
```

### 4.5 PersistenceManager (`src/python/vecdb/persistence.py`)

Handles all disk I/O operations.

```python
class PersistenceManager:
    def __init__(self, db_path: str)
    def save_collection(self, collection: Collection) -> None
    def load_collection(self, name: str) -> Optional[Collection]
    def delete_collection(self, name: str) -> bool
    def list_collections(self) -> List[str]
    def save_metadata(self, metadata: dict) -> None
    def load_metadata(self) -> dict
```

---

## 5. HNSW Algorithm

### 5.1 Overview

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor algorithm that achieves O(log N) search complexity.

### 5.2 Key Concepts

```
Layer 3:  [A] ─────────────────────────── [B]          (fewest nodes)
           │                               │
Layer 2:  [A] ──── [C] ──── [D] ──────── [B]
           │        │        │            │
Layer 1:  [A] ─ [E] [C] ─ [F] [D] ─ [G] ─ [B]
           │    │    │    │    │    │     │
Layer 0:  [A]─[E]─[H]─[C]─[F]─[I]─[D]─[G]─[J]─[B]      (all nodes)
```

### 5.3 Implementation Details

**Graph Structure** (`src/cpp/hnsw_index.hpp:288-291`):
```cpp
std::unordered_map<int64_t, std::vector<float>> vectors_;           // ID → vector
std::unordered_map<int64_t, int> levels_;                           // ID → max level
std::unordered_map<int64_t, std::vector<std::vector<int64_t>>> neighbors_;  // ID → neighbors per level
std::unordered_set<int64_t> deleted_;                               // Deleted IDs
```

**Insert Algorithm** (`src/cpp/hnsw_index.hpp:39-86`):
1. Assign random level via exponential distribution
2. Traverse from top layer to insertion level (greedy)
3. At each level: find neighbors, create bidirectional connections
4. Prune connections if exceeding M limit

**Search Algorithm** (`src/cpp/hnsw_index.hpp:88-108`):
1. Start at entry point (highest level node)
2. Greedy descent: find closest node at each layer
3. At layer 0: expand search with ef_search candidates
4. Return top-k results

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| M | 16 | Max connections per node per layer |
| M_max0 | 32 | Max connections at layer 0 |
| ef_construction | 200 | Search width during build |
| ef_search | 50 | Search width during query |

### 5.4 Level Assignment

```cpp
int random_level() {
    // Exponential distribution: P(level = l) = (1/M)^l
    return static_cast<int>(-std::log(uniform_random()) * (1.0 / std::log(M)));
}
```

---

## 6. Persistence Strategy

### 6.1 File Format

**metadata.json** (Database level):
```json
{
    "version": 1,
    "collections": ["collection1", "collection2"]
}
```

**{name}.meta** (Collection metadata):
```json
{
    "version": 1,
    "name": "my_collection",
    "dimension": 128,
    "metric": "cosine",
    "count": 1000,
    "next_internal_id": 1000
}
```

**{name}.hnsw** (Binary HNSW index):
```
[version: u32][dimension: u64][metric_len: u32][metric: bytes]
[M: u64][ef_construction: u64][entry_point: i64][max_level: i32]
[num_vectors: u64]
For each vector:
    [id: i64][level: i32][vector: f32 * dim][is_deleted: u8]
    For each level:
        [num_neighbors: u32][neighbor_ids: i64 * num_neighbors]
```

**{name}.vectors** (Binary vectors + ID mappings):
```
Header: [version: u32][count: u64][dimension: u32]
Per vector: [user_id: u64][internal_id: u64][floats: f32 * dim]
```

### 6.2 Atomic Writes

To prevent corruption, all writes use atomic operations:

```python
def _atomic_write(self, path: Path, data: bytes):
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.rename(path)  # Atomic on POSIX
```

---

## 7. API Reference

### 7.1 Python API

```python
from vecdb import VecDB, Collection

# Database operations
db = VecDB('./data')                    # Open/create database
db.create_collection('name', dim=128)   # Create collection
db.get_collection('name')               # Get collection
db.delete_collection('name')            # Delete collection
db.list_collections()                   # List all collections
db.save()                               # Save to disk
db.close()                              # Save and close

# Collection operations
col.insert(user_id, vector)             # Insert vector
col.search(query, k=10)                 # Search k-NN
col.get(user_id)                        # Get vector by ID
col.delete(user_id)                     # Delete vector
col.contains(user_id)                   # Check existence
col.count()                             # Count vectors

# Context manager
with VecDB('./data') as db:
    # Auto-saves on exit
    pass
```

### 7.2 Exceptions

| Exception | When Raised |
|-----------|-------------|
| `VecDBError` | Base class for all errors |
| `DimensionError` | Vector dimension mismatch |
| `DuplicateIDError` | Inserting existing ID |
| `CollectionExistsError` | Creating duplicate collection |
| `CollectionNotFoundError` | Accessing non-existent collection |
| `DeserializationError` | Corrupt data on load |

---

## 8. Trade-offs & Future Improvements

### 8.1 Current Trade-offs

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| **In-memory index** | Limited by RAM | Simplicity for MVP; faster operations |
| **Lazy deletion** | Wastes space until rebuild | O(1) delete; avoids graph reconstruction |
| **Single-process** | No concurrent access | Simpler implementation; no locking |
| **Approximate search** | Not 100% recall | O(log N) vs O(N) is worth the trade-off |
| **Python + C++** | Build complexity | Python for usability, C++ for performance |

### 8.2 Known Limitations

1. **No metadata filtering** - Vectors only, no attached attributes
2. **No concurrent writes** - Single-process access assumed
3. **Memory-bound** - Index must fit in RAM
4. **Recall decreases at scale** - 98% at 1K, lower at larger scales
5. **Insert slower than mock** - Graph construction overhead

### 8.3 Future Improvements

| Priority | Improvement | Benefit |
|----------|-------------|---------|
| High | **Increase ef_construction** | Better recall at scale |
| High | **Implement HNSW pruning heuristics** | Improved graph quality |
| Medium | **Add metadata support** | Filter search by attributes |
| Medium | **Memory-mapped files** | Handle larger-than-RAM datasets |
| Medium | **Concurrent reads** | Multi-threaded search |
| Low | **REST API server** | Remote access |
| Low | **Incremental persistence** | Faster saves |
| Low | **SIMD distance functions** | 2-4x faster distance computation |

### 8.4 Benchmark Results

| Vectors | Search Latency | Recall@10 | Notes |
|---------|----------------|-----------|-------|
| 1,000 | 0.97ms | 98% | Meets 95% target |
| 10,000 | 2.03ms | 67% | Needs tuning |

**To improve recall:** Increase `ef_construction` (e.g., 400) and `ef_search` (e.g., 100).

---

## Appendix: File Structure

```
vecdb/
├── CMakeLists.txt                 # Top-level CMake
├── README.md                      # User documentation
├── setup.py                       # Python package setup
├── pyproject.toml                 # Python project config
│
├── src/
│   ├── cpp/                       # C++ HNSW implementation
│   │   ├── CMakeLists.txt
│   │   ├── distance.hpp           # Distance functions
│   │   ├── hnsw_index.hpp         # HNSW algorithm
│   │   └── bindings.cpp           # pybind11 Python bindings
│   │
│   └── python/vecdb/              # Python package
│       ├── __init__.py            # Package exports
│       ├── _version.py            # Version info
│       ├── vecdb.py               # VecDB class (API layer)
│       ├── collection.py          # Collection class
│       ├── persistence.py         # PersistenceManager
│       ├── exceptions.py          # Custom exceptions
│       └── _hnsw_mock.py          # Mock HNSW (testing)
│
├── tests/
│   └── python/
│       ├── test_hnsw_mock.py      # HNSW unit tests
│       ├── test_collection.py     # Collection tests
│       ├── test_persistence.py    # Persistence tests
│       ├── test_vecdb.py          # API tests
│       └── test_integration.py    # Integration tests
│
├── benchmarks/
│   └── benchmark_search.py        # Performance benchmarks
│
├── examples/
│   ├── 01_basic_usage.py
│   ├── 02_distance_metrics.py
│   ├── 03_text_embeddings.py
│   ├── 04_multiple_collections.py
│   ├── 05_crud_operations.py
│   └── 06_numpy_integration.py
│
└── docs/
    ├── PRD.md                     # Product requirements
    ├── ORCHESTRATION.md           # Development workflow
    ├── DESIGN.md                  # This document
    └── BENCHMARKS.md              # Benchmark results
```

---

*End of Design Document*
