#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.h"
#include "muhelper.h"
#include "datatypes.hpp"

#include "binding_helper.h"
#include "binding_tests.hpp"

#define REGISTER_STRUCT(name, ...) py::class_<name>(m, #name).def(py::init<__VA_ARGS__>())

namespace py = pybind11;

PYBIND11_MODULE(disagmoe_c, m) {
    py::class_<MuHelper, std::shared_ptr<MuHelper>>(m, "MuHelper")
        .def("start", &MuHelper::start)
        .def("terminate", &MuHelper::terminate);

    // py::class_<MuAttnDispatcher, std::shared_ptr<MuAttnDispatcher>>(m, "MuAttnDispatcher")
    //     .def(py::init<std::vector<int>, int>())
    //     .def("start", &MuAttnDispatcher::start)
    //     .def("terminate", &MuAttnDispatcher::terminate)
    //     .def("put", &MuAttnDispatcher::put, py::arg("TensorBatch"));

    py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def("wait_for_new_requests", &Scheduler::wait_for_new_requests)
        .def("schedule", &Scheduler::schedule);

    py::class_<MuDispatcher, std::shared_ptr<MuDispatcher>>(m, "MuDispatcher")
        .def("put", &MuDispatcher::put);

    py::class_<TensorBatch>(m, "TensorBatch")
        .def(py::init<>())
        .def_readwrite("data", &TensorBatch::data)
        .def_readwrite("metadata", &TensorBatch::metadata);

    py::class_<ChannelInfo>(m, "ChannelInfo")
        .def(py::init<const std::vector<int> &, const std::vector<int> &>())
        .def_readwrite("expert_ids", &ChannelInfo::expert_ids)
        .def_readwrite("attn_layer_ids", &ChannelInfo::attn_layer_ids);

    py::class_<Channel, std::shared_ptr<Channel>>(m, "Channel");

    REGISTER_STRUCT(TokenMetadata);

    REGISTER_STRUCT(Metadata, std::vector<size_t>);

    py::class_<NcclChannel, Channel, std::shared_ptr<NcclChannel>>(m, "NcclChannel")
        .def("send", &NcclChannel::send)
        .def("recv", &NcclChannel::recv)
        .def("instantiate", &NcclChannel::instantiate);

    // static function calls
    m.def("create_channel", &create_channel);
    m.def("create_channel_py_map", [](int local, int peer, std::map<int, std::string> &uids) {
        puts("converting uuid");
        printf("uid len: %d\n", uids.at(peer).size());
        return create_channel(local, peer, (void*) uids.at(peer).c_str());
    });
    m.def("create_channel_py_single", [](int local, int peer, char* uid) {
        puts("converting uuid");
        return create_channel(local, peer, (void*) uid);
    });
    m.def("get_nccl_unique_id", &get_nccl_unique_id);
    m.def("instantiate_channels", &instantiate_channels);
    m.def("init_engine", &init_engine);
    m.def("start_engine", &start_engine);

    m.def("test_nccl_p2p", &test_nccl_p2p);
    m.def("test_zmq_sub_pub", &test_zmq_sub_pub);
    m.def("test_attn_dispatcher", &test_attn_dispatcher);
    m.def("test_expert_dispatcher", &test_expert_dispatcher);
    m.def("test_scheduler", &test_scheduler);
}