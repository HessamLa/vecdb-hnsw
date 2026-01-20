# C++ Source Directory

This directory contains the C++ implementation of the HNSW index.

## Files (to be created by hnsw_specialist agent)

- `hnsw_index.hpp` - HNSW index class declaration
- `hnsw_index.cpp` - HNSW index implementation
- `distance.hpp` - Distance function declarations
- `distance.cpp` - Distance function implementations
- `bindings.cpp` - pybind11 Python bindings
- `CMakeLists.txt` - Build configuration

## Building

```bash
cd /workspaces/vecdb/build
cmake ..
make -j$(nproc)
```

The compiled module will be available as `vecdb/_hnsw.cpython-*.so`.
