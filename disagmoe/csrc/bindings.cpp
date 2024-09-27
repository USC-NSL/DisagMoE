#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "muhelper.h"
#include "datatypes.h"

#include "binding_helper.h"

namespace py = pybind11;

PYBIND11_MODULE(disagmoe_c, m) {
    py::class_<MuHelper>(m, "MuHelper")
        .def("start", &MuHelper::start)
        .def("terminate", &MuHelper::terminate);

    py::class_<MuAttnDispatcher>(m, "MuAttnDispatcher")
        .def(py::init<std::vector<int>, int>()) // Constructor
        .def("start", &MuAttnDispatcher::start)
        .def("terminate", &MuAttnDispatcher::terminate)
        .def("put", &MuAttnDispatcher::put, py::arg("TensorBatch"));

    py::class_<TensorBatch>(m, "TensorBatch")
        .def(py::init<>())
        .def_readwrite("data", &TensorBatch::data)
        .def_readwrite("metadata", &TensorBatch::metadata);
}