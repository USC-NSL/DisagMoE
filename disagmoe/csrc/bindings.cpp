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

    py::class_<Tokenizer, std::shared_ptr<Tokenizer>>(m, "Tokenizer")
        .def("put_request", &Tokenizer::put_request)
        .def("start", &Tokenizer::start);

    py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler")
        .def("start", &Sampler::start);

    REGISTER_STRUCT(TensorBatch)
        .def_readwrite("data", &TensorBatch::data)
        .def_readwrite("metadata", &TensorBatch::metadata);

    py::class_<ChannelInfo>(m, "ChannelInfo")
        .def(py::init<const std::vector<int> &, const std::vector<int> &>())
        .def_readwrite("expert_ids", &ChannelInfo::expert_ids)
        .def_readwrite("attn_layer_ids", &ChannelInfo::attn_layer_ids);

    py::class_<Channel, std::shared_ptr<Channel>>(m, "Channel");

    REGISTER_STRUCT(TokenMetadata);

    py::class_<Metadata, std::shared_ptr<Metadata>>(m, "Metadata")
        .def(py::init<std::vector<size_t>>())
        .def_readwrite("shape", &Metadata::shape)
        .def_readwrite("dtype", &Metadata::dtype)
        .def_readwrite("layer_id", &Metadata::layer_id)
        .def_readwrite("infos", &Metadata::infos) 
        .def_readwrite("prompt_lens", &Metadata::prompt_lens)
        .def("step_layer", &Metadata::step_layer)
        .def("update_exp_ids", &Metadata::update_exp_ids);

    py::class_<NcclChannel, Channel, std::shared_ptr<NcclChannel>>(m, "NcclChannel")
        .def("send", &NcclChannel::send)
        .def("recv", &NcclChannel::recv)
        .def("instantiate", &NcclChannel::instantiate);

    // static function calls
    m.def("create_channel", &create_channel);
    m.def("create_channel_py_map", [](int local, int peer, std::map<int, std::string> &uids) {
        return create_channel(local, peer, (void*) uids.at(peer).c_str());
    });
    m.def("create_channel_py_single", [](int local, int peer, char* uid) {
        return create_channel(local, peer, (void*) uid);
    });
    m.def("get_nccl_unique_id", &get_nccl_unique_id);
    m.def("instantiate_channels", &instantiate_channels);
    m.def("init_engine", &init_engine);
    m.def("start_engine", &start_engine);
    m.def("init_sampler", &init_sampler);
    m.def("init_tokenizer", &init_tokenizer);

    /********
        Test functions
    ********/
    m.def("test_nccl_p2p", &test_nccl_p2p);
    m.def("test_zmq_sub_pub", &test_zmq_sub_pub);
    m.def("test_attn_dispatcher", &test_attn_dispatcher);
    m.def("test_expert_dispatcher", &test_expert_dispatcher);
    m.def("test_scheduler", &test_scheduler);
    m.def("test_sampler_recv", &test_sampler_recv);
    m.def("test_sampler_send", &test_sampler_send);
    m.def("test_tensor_address", [](uintptr_t data) {
        
    });
}