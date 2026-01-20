cmake_minimum_required(VERSION 3.15)

# Get Python extension suffix
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Create the pybind11 module with full HNSW implementation
pybind11_add_module(_hnsw_cpp bindings.cpp)

# Set output name and directory
set_target_properties(_hnsw_cpp PROPERTIES
    OUTPUT_NAME "_hnsw_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python/vecdb"
)

# Install target (optional, for pip install)
install(TARGETS _hnsw_cpp LIBRARY DESTINATION vecdb)
