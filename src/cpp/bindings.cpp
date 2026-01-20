#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hnsw_index.hpp"

namespace py = pybind11;
using namespace vecdb;

// Convert Python list/numpy to vector<float>
std::vector<float> to_vector(py::object obj, size_t expected_dim) {
    std::vector<float> vec;
    if (py::isinstance<py::array>(obj)) {
        auto arr = obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        vec.assign(arr.data(), arr.data() + arr.size());
    } else {
        vec = obj.cast<std::vector<float>>();
    }
    return vec;
}

PYBIND11_MODULE(_hnsw_cpp, m) {
    m.doc() = "HNSW C++ implementation for VecDB";
    m.attr("__version__") = "0.1.0";
    m.def("is_stub", []() { return false; });

    // Register exceptions
    py::register_exception<DimensionError>(m, "DimensionError", PyExc_ValueError);
    py::register_exception<DuplicateIDError>(m, "DuplicateIDError", PyExc_ValueError);
    py::register_exception<DeserializationError>(m, "DeserializationError", PyExc_ValueError);

    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<size_t, const std::string&, size_t, size_t>(),
             py::arg("dimension"), py::arg("metric"),
             py::arg("M") = 16, py::arg("ef_construction") = 200)
        .def("add", [](HNSWIndex& self, int64_t id, py::object vec) {
            self.add(id, to_vector(vec, self.dimension()));
        }, py::arg("internal_id"), py::arg("vector"))
        .def("search", [](HNSWIndex& self, py::object query, size_t k, size_t ef_search) {
            auto results = self.search(to_vector(query, self.dimension()), k, ef_search);
            py::list ret;
            for (auto& [id, dist] : results) ret.append(py::make_tuple(id, dist));
            return ret;
        }, py::arg("query"), py::arg("k"), py::arg("ef_search") = 50)
        .def("remove", &HNSWIndex::remove, py::arg("internal_id"))
        .def("serialize", [](const HNSWIndex& self) {
            auto data = self.serialize();
            return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
        })
        .def_static("deserialize", [](py::bytes data) {
            std::string s = data;
            std::vector<uint8_t> vec(s.begin(), s.end());
            return HNSWIndex::deserialize(vec);
        }, py::arg("data"))
        .def("__len__", &HNSWIndex::count)
        .def_property_readonly("dimension", &HNSWIndex::dimension)
        .def_property_readonly("metric", &HNSWIndex::metric)
        .def_property_readonly("M", &HNSWIndex::M)
        .def_property_readonly("ef_construction", &HNSWIndex::ef_construction);
}
