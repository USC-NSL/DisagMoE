#pragma once

#include "muhelper.h"
#include "comm.h"

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

class PyChannel: Channel {
public:
    using Channel::Channel;

    void instantiate() override {
        PYBIND11_OVERRIDE_PURE(void, Channel, instantiate);
    }

    void send(uintptr_t data, const Metadata& metadata) override {
        PYBIND11_OVERRIDE_PURE(void, Channel, send);
    }
    
    void recv(uintptr_t data, const Metadata& metadata) override {
        PYBIND11_OVERRIDE_PURE(void, Channel, recv);
    }
};