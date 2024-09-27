#include "muhelper.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class PyMuHelper: MuHelper {
public:
    using MuHelper::MuHelper;
    using MuHelper::start;
    using MuHelper::terminate;

    void run() override {
        PYBIND11_OVERRIDE_PURE(void, MuHelper, run);
    }
};