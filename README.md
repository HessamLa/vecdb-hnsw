# VecDB

Minimal Viable Vector Database with HNSW indexing.

## Quick Start

```python
from vecdb import VecDB

# Create database
with VecDB('./my_database') as db:
    # Create collection
    collection = db.create_collection(
        name='embeddings',
        dimension=384,
        metric='cosine'
    )
    
    # Insert vectors
    collection.insert(user_id=1, vector=[0.1, 0.2, ...])
    
    # Search
    results = collection.search(query=[0.15, 0.25, ...], k=5)
```

## Development Setup

See [docs/SETUP.md](docs/SETUP.md) for development environment setup.

## Documentation

- [Product Requirements](docs/PRD.md)
- [Development Orchestration](docs/ORCHESTRATION.md)
- [Setup Guide](docs/SETUP.md)

## License

MIT
