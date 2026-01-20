/**
 * VecDB HNSW C++ Module - Phase 1 Stub
 *
 * This is a minimal stub module that compiles and imports correctly.
 * The full HNSW implementation will be added in Phase 2.
 *
 * During Phase 1, the Python code uses _hnsw_mock.py instead.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_hnsw_cpp, m) {
    m.doc() = "HNSW C++ module - stub for Phase 1, full implementation in Phase 2";

    // Placeholder function to verify module is loaded
    m.def("is_stub", []() { return true; },
          "Returns true if this is the stub module (Phase 1)");

    // Version info
    m.attr("__version__") = "0.1.0";
}
