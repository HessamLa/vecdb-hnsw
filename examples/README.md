# VecDB Examples

This directory contains example scripts demonstrating VecDB usage.

## Running Examples

```bash
# From the project root
python examples/01_basic_usage.py
python examples/02_distance_metrics.py
# ... etc
```

## Examples Overview

| Example | Description |
|---------|-------------|
| [01_basic_usage.py](01_basic_usage.py) | Getting started: create, insert, search, persist |
| [02_distance_metrics.py](02_distance_metrics.py) | L2, Cosine, and Dot Product metrics explained |
| [03_text_embeddings.py](03_text_embeddings.py) | Simulated text embedding search use case |
| [04_multiple_collections.py](04_multiple_collections.py) | Managing multiple collections |
| [05_crud_operations.py](05_crud_operations.py) | Create, Read, Update, Delete operations |
| [06_numpy_integration.py](06_numpy_integration.py) | Using VecDB with NumPy arrays |

## Quick Reference

### Create a Database

```python
from vecdb import VecDB

# Using context manager (recommended - auto-saves)
with VecDB('./my_database') as db:
    # work with db
    pass

# Or manual management
db = VecDB('./my_database')
# ... work with db ...
db.save()
db.close()
```

### Create a Collection

```python
collection = db.create_collection(
    name='my_vectors',
    dimension=384,          # Vector dimension (required)
    metric='cosine',        # 'l2', 'cosine', or 'dot'
    hnsw_params={           # Optional HNSW tuning
        'M': 16,            # Max connections per node
        'ef_construction': 200  # Build-time search width
    }
)
```

### Insert Vectors

```python
# With list
collection.insert(user_id=1, vector=[0.1, 0.2, 0.3, ...])

# With NumPy array
import numpy as np
collection.insert(user_id=2, vector=np.array([0.1, 0.2, 0.3, ...]))
```

### Search

```python
results = collection.search(
    query=[0.15, 0.25, ...],  # Query vector
    k=10,                      # Number of results
    ef_search=50               # Search-time width (higher = more accurate)
)

for user_id, distance in results:
    print(f"ID: {user_id}, Distance: {distance}")
```

### Other Operations

```python
# Get a vector by ID
vector = collection.get(user_id=1)  # Returns list or None

# Check if ID exists
exists = collection.contains(user_id=1)  # or: 1 in collection

# Delete a vector
deleted = collection.delete(user_id=1)  # Returns True/False

# Count vectors
count = collection.count()  # or: len(collection)
```

### Distance Metrics

| Metric | When to Use | Distance Range |
|--------|-------------|----------------|
| `l2` | General purpose, spatial data | [0, ∞) |
| `cosine` | Text embeddings, normalized vectors | [0, 2] |
| `dot` | Recommendation systems (MIPS) | (-∞, ∞) |

## Tips

1. **Use context manager** - Ensures data is saved on exit
2. **Choose the right metric** - Cosine for text, L2 for spatial, Dot for recommendations
3. **Tune HNSW parameters** - Higher `ef_construction` = better recall, slower build
4. **Normalize for cosine** - Pre-normalize vectors for consistent results
5. **Use NumPy** - Efficient for batch operations
