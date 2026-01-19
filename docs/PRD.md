# Product Requirements Document
## VecDB: Minimal Viable Vector Database

**Version 1.0 | MVP Release**  
**January 2026**

---

# 1. Project Overview

## 1.1 Executive Summary

VecDB is a minimal viable vector database designed to store, index, and query high-dimensional vector embeddings. The system demonstrates core vector database principles with a focus on efficient similarity search using the HNSW (Hierarchical Navigable Small World) algorithm.

## 1.2 Goals

- Implement a functional vector database with CRUD operations for vector data
- Achieve sub-linear search complexity using HNSW indexing (target: O(log N))
- Provide disk persistence for data durability across restarts
- Deliver clean, well-documented code with comprehensive tests
- Support multiple named collections with configurable dimensions

## 1.3 Non-Goals (Out of MVP Scope)

- Metadata storage and filtering
- OLTP/OLAP access pattern optimization
- Concurrency and transaction support
- Distributed operation or replication
- Authentication or access control

## 1.4 Key Design Decisions

| Decision | Choice & Rationale |
|----------|-------------------|
| **Vector Dimensionality** | Fixed per collection. Industry standard; enables SIMD optimization; prevents meaningless cross-dimension comparisons. |
| **ID System** | User-provided uint64. Simplifies C++/Python boundary; fast integer hashing; users maintain business ID mapping. |
| **Collections** | Multiple named collections. Supports different embedding models; logical separation of concerns. |
| **Distance Metrics** | L2, Cosine, Dot Product in C++. Performance-critical hot path; SIMD optimization opportunity. |
| **Architecture** | Library pattern (not client-server). Simpler for MVP; easier testing; server wrapper can be added later. |
| **Cosine Similarity** | Normalize vectors at insert time. Standard optimization; avoids repeated norm computation during search. |

---

# 2. System Architecture

## 2.1 High-Level Architecture

The system follows a layered library architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (Python)                       │
│              VecDB class - Public Interface                 │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Collection Mgr  │  │   HNSW Index    │  │   Persistence   │
│    (Python)     │◄─►│     (C++)       │◄─►│    (Python)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 2.2 Module Overview

| Module | Language | Responsibility |
|--------|----------|----------------|
| **HNSW Index** | C++ | Core HNSW algorithm: graph construction, neighbor search, serialization |
| **Collection Manager** | Python | Collection lifecycle, ID-to-index mapping, dimension validation |
| **Persistence** | Python | Serialize/deserialize collections and metadata to disk |
| **API Layer** | Python | Public VecDB interface, orchestrates all modules |

## 2.3 Data Flow

**Insert Operation:** API validates input → Collection Manager assigns internal index → HNSW builds graph edges → Persistence marks dirty.

**Search Operation:** API validates query vector → HNSW performs greedy search through graph layers → returns top-k internal indices → Collection Manager maps to user IDs.

**Save Operation:** Persistence serializes HNSW graph state and Collection Manager mappings to disk.

---

# 3. Module Specifications

## 3.1 Module: HNSW Index (C++)

### 3.1.1 Purpose

Implements the Hierarchical Navigable Small World algorithm for approximate nearest neighbor search. This is the performance-critical core of the system.

### 3.1.2 Interface Definition

```python
class HNSWIndex:
    def __init__(self, dimension: int, metric: str,
                 M: int = 16, ef_construction: int = 200) -> None
        """
        Initialize empty HNSW index.
        Args:
            dimension: Vector dimensionality (fixed for this index)
            metric: Distance metric - 'l2', 'cosine', or 'dot'
            M: Max connections per node per layer (default: 16)
            ef_construction: Search width during construction (default: 200)
        """

    def add(self, internal_id: int, vector: List[float]) -> None
        """Add vector with given internal ID. Raises if ID exists."""

    def search(self, query: List[float], k: int,
               ef_search: int = 50) -> List[Tuple[int, float]]
        """
        Find k nearest neighbors.
        Returns: List of (internal_id, distance) sorted by distance ascending.
        """

    def remove(self, internal_id: int) -> bool
        """Mark vector as deleted (lazy deletion). Returns success."""

    def serialize(self) -> bytes
        """Serialize entire index to bytes for persistence."""

    @staticmethod
    def deserialize(data: bytes) -> 'HNSWIndex'
        """Reconstruct index from serialized bytes."""

    def __len__(self) -> int
        """Return count of non-deleted vectors."""
```

### 3.1.3 Input/Output Specification

| Method | Input | Output | Errors |
|--------|-------|--------|--------|
| `__init__` | dim (int), metric (str), M (int), ef_construction (int) | None | ValueError if dim < 1 or invalid metric |
| `add` | internal_id (uint64), vector (float[]) | None | DimensionError if len(vector) != dim; DuplicateIDError if ID exists |
| `search` | query (float[]), k (int), ef_search (int) | List[(id, dist)] | DimensionError if len(query) != dim; ValueError if k < 1 |
| `remove` | internal_id (uint64) | bool | None (returns False if not found) |
| `serialize` | None | bytes | None |
| `deserialize` | bytes | HNSWIndex | DeserializationError if corrupt |

### 3.1.4 Success Criteria

- Search returns correct nearest neighbors with >95% recall at k=10 on standard benchmarks
- Search complexity is O(log N) empirically (measure and document)
- Handles at least 100,000 vectors of dimension 128 without memory issues
- Serialization/deserialization is lossless (identical search results before and after)
- All three distance metrics produce mathematically correct results

### 3.1.5 Implementation Notes for Developer

- Use pybind11 for Python bindings
- Implement distance functions with SIMD intrinsics if possible (SSE/AVX)
- For cosine metric: normalize vectors during add(), then use dot product internally
- Use lazy deletion (mark deleted, exclude from results) rather than graph reconstruction
- Entry point should be the node with highest layer
- Serialize format: header (dim, metric, M, ef_c, count) + node data + edge lists

---

## 3.2 Module: Collection Manager (Python)

### 3.2.1 Purpose

Manages the mapping between user-provided IDs and internal HNSW indices, handles collection metadata, and validates inputs before passing to HNSW.

### 3.2.2 Interface Definition

```python
class Collection:
    def __init__(self, name: str, dimension: int, metric: str,
                 hnsw_params: dict = None) -> None
        """
        Create a new collection.
        Args:
            name: Unique collection identifier
            dimension: Fixed vector dimension for this collection
            metric: 'l2', 'cosine', or 'dot'
            hnsw_params: Optional dict with 'M' and 'ef_construction'
        """

    def insert(self, user_id: int, vector: List[float]) -> None
        """Insert vector with user-provided ID."""

    def search(self, query: List[float], k: int = 10,
               ef_search: int = 50) -> List[Tuple[int, float]]
        """Search and return List[(user_id, distance)]."""

    def delete(self, user_id: int) -> bool
        """Delete vector by user ID. Returns success."""

    def get(self, user_id: int) -> Optional[List[float]]
        """Retrieve vector by user ID. Returns None if not found."""

    def contains(self, user_id: int) -> bool
        """Check if user ID exists in collection."""

    def count(self) -> int
        """Return number of vectors in collection."""

    @property
    def name(self) -> str

    @property
    def dimension(self) -> int

    @property
    def metric(self) -> str
```

### 3.2.3 Internal Data Structures

- `user_to_internal: Dict[int, int]` - Maps user IDs to HNSW internal indices
- `internal_to_user: Dict[int, int]` - Reverse mapping for search result translation
- `vectors: Dict[int, List[float]]` - Original vectors stored by user ID (for get() and persistence)
- `next_internal_id: int` - Counter for assigning internal IDs
- `hnsw_index: HNSWIndex` - The underlying C++ index instance

### 3.2.4 Success Criteria

- Correct bidirectional ID mapping (user ID in, user ID out)
- Validates dimension before calling HNSW (clear error messages)
- Handles duplicate user ID insertion gracefully (raise or update, document choice)
- Search returns user IDs, not internal IDs
- Stores original vectors for retrieval via get()

---

## 3.3 Module: Persistence (Python)

### 3.3.1 Purpose

Handles serialization and deserialization of all database state to disk, enabling restart recovery.

### 3.3.2 Interface Definition

```python
class PersistenceManager:
    def __init__(self, db_path: str) -> None
        """
        Initialize persistence at given directory path.
        Creates directory if it doesn't exist.
        """

    def save_collection(self, collection: Collection) -> None
        """Save a single collection to disk."""

    def load_collection(self, name: str) -> Optional[Collection]
        """Load collection by name. Returns None if not found."""

    def delete_collection(self, name: str) -> bool
        """Delete collection files from disk. Returns success."""

    def list_collections(self) -> List[str]
        """List all persisted collection names."""

    def save_metadata(self, metadata: dict) -> None
        """Save database-level metadata."""

    def load_metadata(self) -> dict
        """Load database-level metadata."""
```

### 3.3.3 File Structure

```
db_path/
├── metadata.json          # Database-level metadata
├── collections/
│   ├── {name}.hnsw       # Serialized HNSW index (binary)
│   ├── {name}.meta       # Collection metadata (JSON)
│   └── {name}.vectors    # Original vectors + ID mappings (binary)
```

### 3.3.4 File Formats

- **metadata.json:** JSON containing version, creation date, collection list.
- **{name}.meta:** JSON containing dimension, metric, count, HNSW parameters.
- **{name}.hnsw:** Binary blob from HNSWIndex.serialize().
- **{name}.vectors:** Binary format with header + packed vectors and ID mappings.

### 3.3.5 Success Criteria

- Full round-trip: save then load produces identical search results
- Handles missing files gracefully (returns None, not crash)
- Atomic writes where possible (write to temp, then rename)
- Clear error messages for corruption or version mismatch

---

## 3.4 Module: API Layer (Python)

### 3.4.1 Purpose

Provides the public interface for VecDB users. Orchestrates Collection Manager and Persistence modules.

### 3.4.2 Interface Definition

```python
class VecDB:
    def __init__(self, path: str = './vecdb_data') -> None
        """
        Open or create a VecDB at the given path.
        Loads existing collections from disk if present.
        """

    def create_collection(self, name: str, dimension: int,
                          metric: str = 'l2',
                          hnsw_params: dict = None) -> Collection
        """
        Create a new collection.
        Raises CollectionExistsError if name already taken.
        """

    def get_collection(self, name: str) -> Collection
        """Get existing collection. Raises CollectionNotFoundError."""

    def delete_collection(self, name: str) -> bool
        """Delete collection and its persisted data."""

    def list_collections(self) -> List[str]
        """List all collection names."""

    def save(self) -> None
        """Persist all collections to disk."""

    def close(self) -> None
        """Save and release resources."""

    def __enter__(self) -> 'VecDB'

    def __exit__(self, *args) -> None
        """Context manager support for automatic save on exit."""
```

### 3.4.3 Usage Example

```python
from vecdb import VecDB

# Create database
with VecDB('./my_database') as db:
    # Create collection for 384-dim embeddings
    collection = db.create_collection(
        name='documents',
        dimension=384,
        metric='cosine'
    )

    # Insert vectors
    collection.insert(user_id=1001, vector=[0.1, 0.2, ...])
    collection.insert(user_id=1002, vector=[0.3, 0.4, ...])

    # Search
    results = collection.search(query=[0.15, 0.25, ...], k=5)
    for user_id, distance in results:
        print(f'ID: {user_id}, Distance: {distance}')

    # Explicit save (also happens on context exit)
    db.save()

# Later: reopen and data persists
db = VecDB('./my_database')
collection = db.get_collection('documents')
print(collection.count())  # Still has the vectors
```

### 3.4.4 Success Criteria

- Clean, intuitive API that matches the usage example
- Context manager ensures data is saved on exit
- Clear, specific exception types for different error conditions
- Loading existing database restores all collections correctly

---

# 4. Inter-Module Communication

All modules communicate via direct Python function calls. The HNSW C++ module is accessed through pybind11 bindings that expose a Pythonic interface.

## 4.1 Communication Diagram

```
User Code
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  VecDB (API Layer)                                          │
│    │                                                        │
│    ├──► PersistenceManager.load_metadata()                  │
│    │        └──► Returns: dict                              │
│    │                                                        │
│    ├──► Collection.__init__(name, dim, metric)              │
│    │        └──► HNSWIndex.__init__(dim, metric, M, ef_c)   │
│    │                                                        │
│    ├──► Collection.insert(user_id, vector)                  │
│    │        ├──► Validate dimension                         │
│    │        ├──► Assign internal_id                         │
│    │        └──► HNSWIndex.add(internal_id, vector)         │
│    │                                                        │
│    ├──► Collection.search(query, k)                         │
│    │        ├──► HNSWIndex.search(query, k, ef_search)      │
│    │        │        └──► Returns: [(internal_id, dist)]    │
│    │        └──► Map internal_ids to user_ids               │
│    │                                                        │
│    └──► VecDB.save()                                        │
│             ├──► Collection.hnsw_index.serialize()          │
│             │        └──► Returns: bytes                    │
│             └──► PersistenceManager.save_collection()       │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 Data Exchange Formats

| Interface | Data Type | Notes |
|-----------|-----------|-------|
| Python → C++ vectors | List[float] or numpy.ndarray | pybind11 handles conversion automatically |
| C++ → Python results | List[Tuple[int, float]] | List of (internal_id, distance) pairs |
| C++ serialization | bytes (Python) / std::vector<uint8_t> (C++) | Opaque binary blob |
| Collection metadata | JSON-serializable dict | Stored in .meta files |

## 4.3 Error Propagation

C++ exceptions are translated to Python exceptions via pybind11. Custom exception types are defined:

- **DimensionError:** Vector dimension mismatch
- **DuplicateIDError:** Attempting to insert existing ID
- **CollectionExistsError:** Creating collection with existing name
- **CollectionNotFoundError:** Accessing non-existent collection
- **DeserializationError:** Corrupt or incompatible serialized data

---

# 5. Development Guidelines

Each module is to be developed by a separate developer/agent. This section provides specific instructions for each.

## 5.1 HNSW Index Developer (C++)

### 5.1.1 Deliverables

- `hnsw_index.hpp` / `hnsw_index.cpp` - Core implementation
- `distance.hpp` / `distance.cpp` - Distance functions (L2, cosine, dot)
- `bindings.cpp` - pybind11 Python bindings
- `CMakeLists.txt` - Build configuration
- `tests/test_hnsw.cpp` - C++ unit tests

### 5.1.2 Technical Requirements

- C++17 or later
- No external dependencies except pybind11 and standard library
- Use std::vector<float> for vector storage
- Use std::unordered_map for adjacency lists
- Document HNSW parameters (M, ef_construction, ef_search) with sensible defaults

### 5.1.3 HNSW Algorithm Summary

- Multi-layer graph where each layer is a subset of the layer below
- Top layer has fewest nodes, bottom layer has all nodes
- Search: start at entry point in top layer, greedily descend, refine at each layer
- Insert: determine max layer via exponential distribution, connect at each layer up to max
- Each node maintains up to M connections per layer

### 5.1.4 Testing Requirements

- Unit test each distance function for correctness
- Test insertion of 1000+ vectors
- Test search returns correct results (compare to brute force on small dataset)
- Test serialization round-trip
- Test edge cases: empty index, k larger than index size, duplicate IDs

---

## 5.2 Collection Manager Developer (Python)

### 5.2.1 Deliverables

- `collection.py` - Collection class implementation
- `exceptions.py` - Custom exception definitions
- `tests/test_collection.py` - Unit tests

### 5.2.2 Technical Requirements

- Python 3.9+
- Type hints on all public methods
- Depend only on the HNSW module interface (not implementation)
- Store original vectors in memory for get() retrieval

### 5.2.3 Key Behaviors to Implement

- On insert: validate dimension, assign internal_id, store vector, call hnsw.add()
- On search: call hnsw.search(), translate internal IDs to user IDs
- On delete: call hnsw.remove(), update mappings
- Decision: reject duplicate user IDs with DuplicateIDError (simpler than update semantics)

---

## 5.3 Persistence Developer (Python)

### 5.3.1 Deliverables

- `persistence.py` - PersistenceManager class
- `tests/test_persistence.py` - Unit tests

### 5.3.2 Technical Requirements

- Use standard library only (json, struct, pathlib, os)
- Use atomic writes: write to .tmp file, then rename
- Include version number in file headers for future compatibility

### 5.3.3 Serialization Details

For `.vectors` file, use struct module for binary packing:

```
Header: [version: uint32][count: uint64][dimension: uint32]
Per vector: [user_id: uint64][internal_id: uint64][floats: float32 * dim]
```

---

## 5.4 API Layer Developer (Python)

### 5.4.1 Deliverables

- `vecdb.py` - VecDB class (main entry point)
- `__init__.py` - Package exports
- `tests/test_vecdb.py` - Integration tests

### 5.4.2 Technical Requirements

- Orchestrate Collection and Persistence modules
- Load existing collections on __init__
- Implement context manager protocol
- Provide clear docstrings matching the interface specification

### 5.4.3 Integration Responsibilities

- Write integration tests that exercise the full flow: create, insert, search, save, reload
- Document the public API in README.md
- Create usage examples

---

# 6. Package Dependencies

## 6.1 Build Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| cmake | ≥3.15 | C++ build system |
| pybind11 | ≥2.10 | C++ to Python bindings |
| g++ / clang++ | C++17 support | C++ compiler |
| python | ≥3.9 | Runtime and development |

## 6.2 Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20 | Efficient vector handling at Python layer |

## 6.3 Development/Test Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | ≥7.0 | Python test framework |
| pytest-cov | ≥4.0 | Code coverage reporting |

## 6.4 Containerization

For reproducibility, the project will use Docker with the following base image:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pybind11 \
    numpy \
    pytest \
    pytest-cov
```

---

# 7. Testing Strategy

## 7.1 Test Levels

| Level | Scope | Owner |
|-------|-------|-------|
| Unit Tests | Individual functions and classes in isolation | Each module developer |
| Integration Tests | Module interactions (e.g., Collection + HNSW) | API Layer developer |
| End-to-End Tests | Full workflows including persistence | API Layer developer |

## 7.2 Required Test Cases

### 7.2.1 HNSW Module Tests

- `test_distance_l2`: Verify L2 distance calculation
- `test_distance_cosine`: Verify cosine similarity calculation
- `test_distance_dot`: Verify dot product calculation
- `test_add_single`: Add one vector, verify count
- `test_add_multiple`: Add 100 vectors, verify count
- `test_search_exact`: Insert known vectors, verify exact match returned first
- `test_search_recall`: Compare to brute force, measure recall
- `test_remove`: Remove vector, verify excluded from search
- `test_serialize_deserialize`: Round-trip, verify identical results
- `test_dimension_mismatch`: Verify error on wrong dimension
- `test_duplicate_id`: Verify error on duplicate internal ID

### 7.2.2 Collection Manager Tests

- `test_insert_and_get`: Insert vector, retrieve by user ID
- `test_search_returns_user_ids`: Verify search returns user IDs not internal
- `test_delete`: Delete vector, verify not in search results
- `test_duplicate_user_id`: Verify error on duplicate
- `test_dimension_validation`: Verify error on wrong dimension

### 7.2.3 Persistence Tests

- `test_save_load_collection`: Save, load, verify identical
- `test_list_collections`: Save multiple, list returns all names
- `test_delete_collection`: Delete, verify files removed
- `test_missing_file`: Load non-existent, verify returns None
- `test_corrupt_file`: Load corrupt data, verify graceful error

### 7.2.4 Integration Tests

- `test_full_workflow`: Create DB, add collection, insert, search, save, reload, search again
- `test_multiple_collections`: Create two collections with different dimensions
- `test_context_manager`: Use with statement, verify save on exit

---

# 8. Project File Structure

```
vecdb/
├── CMakeLists.txt              # Top-level CMake config
├── Dockerfile                  # Container definition
├── README.md                   # Documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Python package setup
│
├── src/
│   ├── cpp/                    # C++ HNSW implementation
│   │   ├── CMakeLists.txt
│   │   ├── hnsw_index.hpp
│   │   ├── hnsw_index.cpp
│   │   ├── distance.hpp
│   │   ├── distance.cpp
│   │   └── bindings.cpp       # pybind11 bindings
│   │
│   └── python/
│       └── vecdb/             # Python package
│           ├── __init__.py
│           ├── vecdb.py       # VecDB class (API Layer)
│           ├── collection.py  # Collection class
│           ├── persistence.py # PersistenceManager class
│           └── exceptions.py  # Custom exceptions
│
└── tests/
    ├── cpp/
    │   └── test_hnsw.cpp      # C++ unit tests
    │
    └── python/
        ├── test_hnsw.py       # HNSW binding tests
        ├── test_collection.py
        ├── test_persistence.py
        └── test_vecdb.py      # Integration tests
```

---

# 9. Development Timeline

Total duration: 9 days. Modules can be developed in parallel after Day 1.

| Day | Focus | Deliverables |
|-----|-------|--------------|
| 1 | Setup & Planning | Repository structure, Dockerfile, CI/CD setup, PRD finalization |
| 2-4 | HNSW Implementation | Core C++ algorithm, distance functions, pybind11 bindings, C++ tests |
| 3-5 | Collection Manager | Python Collection class, ID mapping, validation, unit tests |
| 4-6 | Persistence | PersistenceManager, file formats, atomic writes, unit tests |
| 5-7 | API Layer | VecDB class, integration with all modules, integration tests |
| 8 | Documentation | README, API reference, design doc, usage examples |
| 9 | Final Testing & Polish | End-to-end tests, benchmarks, code review, cleanup |

---

# 10. Appendix: Distance Metric Formulas

## 10.1 L2 (Euclidean) Distance

```
d(a, b) = sqrt(sum((a[i] - b[i])^2 for i in range(dim)))
```

```cpp
// C++ implementation:
float l2_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
```

## 10.2 Cosine Similarity

```
sim(a, b) = dot(a, b) / (norm(a) * norm(b))
distance = 1 - sim  // Convert similarity to distance
```

```
// Optimization: normalize at insert time, then use dot product
// normalized_a = a / norm(a)
// distance = 1 - dot(normalized_a, normalized_b)
```

## 10.3 Dot Product (Inner Product)

```
dot(a, b) = sum(a[i] * b[i] for i in range(dim))
```

```
// For MIPS (Maximum Inner Product Search), we want maximum dot product
// HNSW typically minimizes, so use negative: distance = -dot(a, b)
```

---

*— End of Document —*