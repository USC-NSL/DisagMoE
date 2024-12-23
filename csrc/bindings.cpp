#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "tests.h"
#include "engine.h"
#include "muhelper.h"
#include "datatypes.hpp"
#include "block_manager.h"
#include "permute.h"
#include "binding_helper.h"
#include "binding_tests.hpp"

#define REGISTER_STRUCT(name, ...) py::class_<name>(m, #name).def(py::init<__VA_ARGS__>())
#define REGISTER_FUNC(name) m.def(#name, &name)

PYBIND11_MAKE_OPAQUE(std::map<std::pair<int, int>, int>);

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
        .def("schedule", &Scheduler::schedule)
        .def("get_pool_snapshot", &Scheduler::get_pool_snapshot);

    py::class_<AttentionScheduler, attn_scheduler_t>(m, "AttentionScheduler")
        .def("wait_for_new_requests", &AttentionScheduler::wait_for_new_requests)
        .def("schedule", &AttentionScheduler::schedule)
        .def("get_channel", &AttentionScheduler::get_channel)
        .def("set_max_batch_size", &AttentionScheduler::set_max_batch_size)
        .def("get_pool_snapshot", &AttentionScheduler::get_pool_snapshot);

    py::class_<MuDispatcher, std::shared_ptr<MuDispatcher>>(m, "MuDispatcher")
        .def("put", &MuDispatcher::put);

    py::class_<Tokenizer, std::shared_ptr<Tokenizer>>(m, "Tokenizer")
        .def("put_request", &Tokenizer::put_request)
        .def("start", &Tokenizer::start);

    py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler")
        .def("start", &Sampler::start)
        .def("wait_slo_stats", &Sampler::wait_slo_stats)
        .def("fetch_finished_slo_stats", &Sampler::fetch_finished_slo_stats);

    REGISTER_STRUCT(TensorBatch)
        .def_readwrite("data", &TensorBatch::data)
        .def_readwrite("metadata", &TensorBatch::metadata);

    py::class_<ChannelInfo>(m, "ChannelInfo")
        .def(py::init<const std::vector<ExpertId> &, const std::vector<int> &, int>())
        .def_readwrite("expert_ids", &ChannelInfo::expert_ids)
        .def_readwrite("attn_layer_ids", &ChannelInfo::attn_layer_ids)
        .def_readwrite("attn_dp_rank", &ChannelInfo::attn_dp_rank);

    py::class_<Channel, std::shared_ptr<Channel>>(m, "Channel");

    py::class_<NcclGroupChannel, std::shared_ptr<NcclGroupChannel>>(m, "NcclGroupChannel")
        .def("all_reduce", &NcclGroupChannel::all_reduce);

    REGISTER_STRUCT(TokenMetadata);

    py::class_<ParallelConfig>(m, "ParallelConfig")
        .def(py::init<>())
        .def_readwrite("tp", &ParallelConfig::tp)
        .def_readwrite("ep", &ParallelConfig::ep)
        .def_readwrite("n_exp_per_rank", &ParallelConfig::n_exp_per_rank)
        .def_readwrite("expert_ranks", &ParallelConfig::expert_ranks);

    REGISTER_STRUCT(AttentionBatch)
        .def_readwrite("data", &AttentionBatch::data)
        .def_readwrite("metadata", &AttentionBatch::metadata);

    REGISTER_STRUCT(SloStat)
        .def_readwrite("req_id", &SloStat::req_id)
        .def_readwrite("t_prefill", &SloStat::t_prefill)
        .def_readwrite("t_decode", &SloStat::t_decode)
        .def_readwrite("t_tokens", &SloStat::t_tokens);

    py::class_<AttentionBatchMetadata, std::shared_ptr<AttentionBatchMetadata>>(m, "AttentionBatchMetadata")
        .def(py::init<>())
        .def_readwrite("shape", &AttentionBatchMetadata::shape)
        .def_readwrite("dtype", &AttentionBatchMetadata::dtype)
        .def_readwrite("layer_id", &AttentionBatchMetadata::layer_id)
        .def_readwrite("seq_ids", &AttentionBatchMetadata::seq_ids)
        .def_readwrite("num_prefill_tokens", &AttentionBatchMetadata::num_prefill_tokens)
        .def_readwrite("num_prefill_seqs", &AttentionBatchMetadata::num_prefill_seqs)
        .def_readwrite("num_decode_tokens", &AttentionBatchMetadata::num_decode_tokens)
        .def_readwrite("prefill_seq_len", &AttentionBatchMetadata::prefill_seq_len)
        .def_readwrite("prefill_query_len", &AttentionBatchMetadata::prefill_query_len)
        .def_readwrite("expert_ids", &AttentionBatchMetadata::expert_ids)
        .def("to_metadata", &AttentionBatchMetadata::to_metadata);

    py::class_<Metadata, std::shared_ptr<Metadata>>(m, "Metadata")
        .def(py::init<std::vector<size_t>>())
        .def_readwrite("shape", &Metadata::shape)
        .def_readwrite("dtype", &Metadata::dtype)
        .def_readwrite("layer_id", &Metadata::layer_id)
        .def_readwrite("req_ids", &Metadata::req_ids)
        .def_readwrite("exp_ids", &Metadata::exp_ids)
        .def_readwrite("prefill_poss", &Metadata::prefill_poss)
        .def("step_layer", &Metadata::step_layer)
        .def("update_exp_ids", &Metadata::update_exp_ids)
        .def("permute_token_infos", &Metadata::permute_token_infos)
        .def("get_expert_batch_sizes", &Metadata::get_expert_batch_sizes)
        .def("sort_by_prefill_order", &Metadata::sort_by_prefill_order);

    py::class_<NcclChannel, Channel, std::shared_ptr<NcclChannel>>(m, "NcclChannel")
        .def("send", &NcclChannel::send)
        .def("recv", &NcclChannel::recv)
        .def("instantiate", &NcclChannel::instantiate);

    py::class_<BlockManager, std::shared_ptr<BlockManager>>(m, "BlockManager")
        .def(py::init<int, int, int>())
        .def("can_allocate", &BlockManager::can_allocate)
        .def("allocate", &BlockManager::allocate)
        .def("release", &BlockManager::release)
        .def("batch_release", &BlockManager::batch_release)
        .def("can_append", &BlockManager::can_append)
        .def("append_block", &BlockManager::append_block)
        .def("num_free_blocks", &BlockManager::num_free_blocks)
        .def("has_seq_block_list", &BlockManager::has_seq_block_list)
        .def("append_tokens", &BlockManager::append_tokens)
        .def("update_block_table", &BlockManager::update_block_table)
        .def("prepare_block_table", &BlockManager::prepare_block_table);

    // custom ops
    m.def("permute_tokens_cuda", &permute_tokens_cuda);

    // static function calls
    m.def("create_channel", &create_channel);
    m.def("create_nccl_group_channel", &create_nccl_group_channel);
    m.def("create_nccl_group_channels", &create_nccl_group_channels);
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
    REGISTER_FUNC(set_hosts);

    /********
        Test functions
    ********/
    m.def("test_nccl_p2p", &test_nccl_p2p);
    m.def("test_nccl_group", &test_nccl_group);
    m.def("test_parallel_attn_scheduler", &test_parallel_attn_scheduler);
    m.def("test_multi_launch", &test_multi_launch);

    REGISTER_FUNC(test_op_overlap);
    // m.def("test_zmq_sub_pub", &test_zmq_sub_pub);
    // m.def("test_attn_dispatcher", &test_attn_dispatcher);
    // m.def("test_expert_dispatcher", &test_expert_dispatcher);
    // m.def("test_scheduler", &test_scheduler);
    // m.def("test_sampler_recv", &test_sampler_recv);
    // m.def("test_sampler_send", &test_sampler_send);
}