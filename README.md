# VecDB

A minimal vector database with HNSW (Hierarchical Navigable Small World) indexing for efficient similarity search.

## Features

- **Multiple Collections**: Named collections with configurable dimensions
- **Distance Metrics**: L2 (Euclidean), Cosine, Dot Product
- **Persistence**: Automatic save/load from disk
- **HNSW Indexing**: O(log N) approximate nearest neighbor search
- **Python API**: Clean, intuitive interface

## Quick Start

```python
from vecdb import VecDB

# Create database (auto-saves on context exit)
with VecDB('./my_database') as db:
    # Create a collection for 384-dimensional embeddings
    collection = db.create_collection(
        name='documents',
        dimension=384,
        metric='cosine'
    )

    # Insert vectors with user-provided IDs
    collection.insert(user_id=1001, vector=[0.1, 0.2, ...])
    collection.insert(user_id=1002, vector=[0.3, 0.4, ...])

    # Search for similar vectors
    results = collection.search(query=[0.15, 0.25, ...], k=5)
    for user_id, distance in results:
        print(f"ID: {user_id}, Distance: {distance:.4f}")

    # Retrieve a vector
    vec = collection.get(1001)  # Returns the original vector

    # Delete a vector
    collection.delete(1002)

# Data persists across sessions
db = VecDB('./my_database')
collection = db.get_collection('documents')
print(f"Collection has {collection.count()} vectors")
```

## Installation

### Docker (Recommended)

```bash
docker-compose up -d
docker exec -it vecdb-dev bash
```

### Local Development

```bash
# Install Python dependencies
pip install -e .

# Build C++ HNSW module
mkdir -p cmake-build && cd cmake-build
cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
```

## API Reference

### VecDB

```python
VecDB(path: str = './vecdb_data')
```

Main database class. Creates/opens a database at the specified path.

**Methods:**
- `create_collection(name, dimension, metric='l2', hnsw_params=None)` → Collection
- `get_collection(name)` → Collection
- `delete_collection(name)` → bool
- `list_collections()` → List[str]
- `save()` → None
- Context manager support (`with VecDB(...) as db:`)

### Collection

```python
Collection(name, dimension, metric, hnsw_params=None)
```

A collection of vectors with a fixed dimension.

**Methods:**
- `insert(user_id: int, vector: List[float])` → None
- `search(query: List[float], k: int = 10, ef_search: int = 50)` → List[Tuple[int, float]]
- `get(user_id: int)` → Optional[List[float]]
- `delete(user_id: int)` → bool
- `contains(user_id: int)` → bool
- `count()` → int

**Properties:**
- `name: str`
- `dimension: int`
- `metric: str` ('l2', 'cosine', or 'dot')

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `l2` | Euclidean distance | General purpose |
| `cosine` | 1 - cosine similarity | Text embeddings |
| `dot` | Negative dot product | Maximum inner product search |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (Python)                       │
│                    VecDB class                              │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Collection Mgr  │  │   HNSW Index    │  │   Persistence   │
│    (Python)     │◄─►│     (C++)       │◄─►│    (Python)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### HNSW Algorithm

The Hierarchical Navigable Small World algorithm provides:
- **Multi-layer graph structure**: Higher layers for coarse navigation, lower layers for fine-grained search
- **O(log N) search complexity**: Sub-linear search time
- **High recall**: >95% recall@10 on standard benchmarks
- **Configurable parameters**: M (connections), ef_construction, ef_search

## Benchmarks

| Vectors | Search Latency | Recall@10 |
|---------|----------------|-----------|
| 1,000   | 0.97ms         | 98%       |
| 10,000  | 2.03ms         | 67%       |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed results.

## File Structure

```
vecdb_data/
├── metadata.json          # Database metadata
└── collections/
    ├── {name}.meta       # Collection metadata (JSON)
    ├── {name}.hnsw       # HNSW index (binary)
    └── {name}.vectors    # Vectors + ID mappings (binary)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=vecdb
```

## Trade-offs & Future Improvements

### Current Limitations
- No metadata filtering (vectors only)
- Single-process access (no concurrent writes)
- In-memory index (limited by RAM)

### Future Improvements
- Metadata support with filtering
- Disk-based index for larger datasets
- Concurrent read/write support
- Server mode with REST API
- Improved recall via better neighbor selection

## Documentation

- [Product Requirements](docs/PRD.md)
- [Development Orchestration](docs/ORCHESTRATION.md)
- [Benchmarks](docs/BENCHMARKS.md)

## License

MIT
