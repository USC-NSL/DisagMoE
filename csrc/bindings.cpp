#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "tests.h"
#include "engine.h"
#include "muhelper.h"
#include "datatypes.hpp"
#include "block_manager.h"
#include "binding_helper.h"
#include "profiler.hpp"
#include "transport_factory.h"
#include "tensor_utils.hpp"
#include "grouped_gemm.h"

#if USE_NIXL
#include "nixl_context.h"
#endif

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
    py::class_<MuPool, std::shared_ptr<MuPool>>(m, "MuPool")
        .def("put_batch", &MuPool::put_batch)
        .def("set_tracing_enabled", &MuPool::set_tracing_enabled)
        .def("drain_recv_completion_stats", &MuPool::drain_recv_completion_stats);
        
    py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def("get_pool_snapshot", &Scheduler::get_pool_snapshot,
             "Return cached pool snapshot from the most recent schedule() call.")
        .def("get_topk_pool_snapshot", &Scheduler::get_topk_pool_snapshot)
        .def("set_schedule_policy", &Scheduler::set_schedule_policy)
        .def("set_schedule_block", &Scheduler::set_schedule_block)
        .def("set_schedule_token_threshold", &Scheduler::set_schedule_token_threshold)
        .def("schedule_trace", &Scheduler::schedule_trace,
             "Schedule next batch and return (batch, pre-schedule snapshot) as ScheduleTrace.")
        .def("schedule", &Scheduler::schedule,
             "Schedule next batch; also refreshes the cached pool snapshot atomically.");

    // Pairs a scheduled batch with the pool snapshot taken in the same scheduling call for tracing purpose.
    py::class_<Scheduler::ScheduleTrace>(m, "ScheduleTrace")
        .def(py::init<>())
        .def_readwrite("batch", &Scheduler::ScheduleTrace::batch)
        .def_readwrite("pool_snapshot", &Scheduler::ScheduleTrace::pool_snapshot);

    py::class_<MuDispatcher, std::shared_ptr<MuDispatcher>>(m, "MuDispatcher")
        .def("put", &MuDispatcher::put)
        .def("set_max_pending_sends", &MuDispatcher::set_max_pending_sends)
        .def("set_tracing_enabled", &MuDispatcher::set_tracing_enabled)
        .def("drain_pending_send_stall_stats", &MuDispatcher::drain_pending_send_stall_stats)
        .def("drain_send_msg_size_stats", &MuDispatcher::drain_send_msg_size_stats);

#if USE_NIXL
    m.def("nixl_set_tracing_enabled", [](bool v) {
        NixlContext::instance().set_tracing_enabled(v);
    });
    m.def("nixl_drain_send_traces", []() {
        return NixlContext::instance().drain_send_traces();
    });
    m.def("nixl_drain_recv_traces", []() {
        return NixlContext::instance().drain_recv_traces();
    });
#else
    m.def("nixl_set_tracing_enabled", [](bool) {});
    m.def("nixl_drain_send_traces", []() {
        return std::vector<std::tuple<int,int,int,size_t,int,double,double,double,double,double,double>>{};
    });
    m.def("nixl_drain_recv_traces", []() {
        return std::vector<std::tuple<int,int,int,size_t,double,double,double,double,double>>{};
    });
#endif

    py::class_<ChannelInfo>(m, "ChannelInfo")
        .def(py::init<const std::vector<ExpertId> &, const std::vector<int> &, int>())
        .def_readwrite("expert_ids", &ChannelInfo::expert_ids)
        .def_readwrite("attn_layer_ids", &ChannelInfo::attn_layer_ids)
        .def_readwrite("attn_dp_rank", &ChannelInfo::attn_dp_rank);

    py::class_<Channel, std::shared_ptr<Channel>>(m, "Channel");

    REGISTER_STRUCT(TokenMetadata);

    py::class_<ParallelConfig>(m, "ParallelConfig")
        .def(py::init<>())
        .def_readwrite("tp", &ParallelConfig::tp)
        .def_readwrite("ep", &ParallelConfig::ep)
        .def_readwrite("dp", &ParallelConfig::dp)
        .def_readwrite("n_exp_per_rank", &ParallelConfig::n_exp_per_rank)
        .def_readwrite("n_total_experts", &ParallelConfig::n_total_experts)
        .def_readwrite("expert_ranks", &ParallelConfig::expert_ranks);
        
    py::class_<BatchMetadata, std::shared_ptr<BatchMetadata>>(m, "BatchMetadata")
        .def(py::init<>())
        .def_readwrite("shape", &BatchMetadata::shape)
        .def_readwrite("dtype", &BatchMetadata::dtype)
        .def_readwrite("layer_id", &BatchMetadata::layer_id)
        .def_readwrite("req_ids", &BatchMetadata::req_ids)
        .def_readwrite("exp_ids", &BatchMetadata::exp_ids)
        .def_readwrite("topk_weights", &BatchMetadata::topk_weights)
        .def_readwrite("attn_dp_ranks", &BatchMetadata::attn_dp_ranks)
        .def_readwrite("init_prefill_lens", &BatchMetadata::init_prefill_lens)
        .def_readwrite("num_prefill_seqs", &BatchMetadata::num_prefill_seqs)
        .def_readwrite("num_prefill_tokens", &BatchMetadata::num_prefill_tokens)
        .def_readwrite("num_decode_tokens", &BatchMetadata::num_decode_tokens)
        .def("is_attention", &BatchMetadata::is_attention)
        .def("is_expert", &BatchMetadata::is_expert)
        .def("is_tokenizer", &BatchMetadata::is_tokenizer)
        .def("num_tokens", &BatchMetadata::num_tokens)
        .def("token_hidden_dim", &BatchMetadata::token_hidden_dim)
        .def("step_layer", &BatchMetadata::step_layer)
        .def("set_finish_signal", &BatchMetadata::set_finish_signal)
        .def("get_expert_batch_sizes", &BatchMetadata::get_expert_batch_sizes)
        .def("get_expert_batch_sizes_cuda", &BatchMetadata::get_expert_batch_sizes_cuda)
        .def("get_token_expert_indices", &BatchMetadata::get_token_expert_indices)
        .def("get_finished_indices", &BatchMetadata::get_finished_indices)
        .def("permute_token_infos", &BatchMetadata::permute_token_infos)
        .def("duplicate_topk", &BatchMetadata::duplicate_topk)
        .def("sort_by_attention", &BatchMetadata::sort_by_attention)
        .def("sort_by_expert", &BatchMetadata::sort_by_expert)
        .def("index_select", &BatchMetadata::index_select);
        
    py::class_<TokenBatch>(m, "TokenBatch")
        .def(py::init<>())
        .def_readwrite("data", &TokenBatch::data)
        .def_readwrite("metadata", &TokenBatch::metadata);

    py::class_<BlockManager, std::shared_ptr<BlockManager>>(m, "BlockManager")
        .def(py::init<int, int, int>())
        .def("close", &BlockManager::close)
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
        .def("prepare_block_table", &BlockManager::prepare_block_table)
        .def("prepare_block_table_gdr", &BlockManager::prepare_block_table_gdr)
        .def("prepare_seq_info", &BlockManager::prepare_seq_info)
        .def("prepare_seq_info_gdr", &BlockManager::prepare_seq_info_gdr);

    py::class_<GdrContext, std::shared_ptr<GdrContext>>(m, "GdrContext")
        .def(py::init<const torch::Tensor&>())
        .def("get_tensor", &GdrContext::get_tensor)
        .def("copy_from_host", &GdrContext::copy_from_host)
        .def("copy_from_host_tensor", &GdrContext::copy_from_host_tensor)
        .def("copy_to_host", &GdrContext::copy_to_host)
        .def("copy_to_host_tensor", &GdrContext::copy_to_host_tensor)
        .def("fill", &GdrContext::fill)
        .def("copy_from_host_int32", &GdrContext::copy_from_host_int32)
        .def("copy_from_host_int64", &GdrContext::copy_from_host_int64)
        .def("copy_from_host_float", &GdrContext::copy_from_host_float)
        .def("copy_to_host_int32", &GdrContext::copy_to_host_int32)
        .def("copy_to_host_float", &GdrContext::copy_to_host_float)
        .def("copy_to_host_int64", &GdrContext::copy_to_host_int64);

    REGISTER_FUNC(rebind_1d_tensor);
    REGISTER_FUNC(rebind_2d_tensor);
    REGISTER_FUNC(rebind_batch_info_tensor);
    
    // profiler functions
    m.def("recorder_output", &Recorder::output);
    m.def("recorder_create", &Recorder::create);
    m.def("range_push", &Recorder::push);
    m.def("range_pop", &Recorder::pop);

    py::class_<TraceContext>(m, "TraceContext")
        .def_readwrite("msg", &TraceContext::msg)
        .def_readwrite("t_start", &TraceContext::t_start)
        .def_readwrite("t_dur", &TraceContext::t_dur)
        .def_readwrite("track_id", &TraceContext::track_id);

    REGISTER_FUNC(get_nccl_unique_id);
    REGISTER_FUNC(init_disaggregated_engine);
    REGISTER_FUNC(init_unified_engine);
    REGISTER_FUNC(start_engine);
    REGISTER_FUNC(set_hosts);

    // Transport selection from Python (required before engine init)
    m.def("select_transport", &disagmoe::select_transport, py::arg("name"));

    // CUTLASS Grouped GEMM for MoE experts (sm < 90, CUTLASS)
    m.def("init_grouped_gemm", &disagmoe::init_grouped_gemm,
          py::arg("device_id"),
          "Probe hardware and select suitable CUTLASS tile config. Returns description string.");

    py::class_<disagmoe::CutlassGemmRunner, std::shared_ptr<disagmoe::CutlassGemmRunner>>(m, "CutlassGemmRunner")
        .def(py::init<torch::Tensor, int64_t>(),
             py::arg("b_weight"), py::arg("max_tokens"),
             "Create a CUTLASS grouped GEMM runner for a weight tensor.")
        .def("setup_meta", &disagmoe::CutlassGemmRunner::setup_meta,
             py::arg("a"), py::arg("c"), py::arg("batch_sizes"),
             "Update CUTLASS metadata arrays on device. Graph-capturable.")
        .def("run", &disagmoe::CutlassGemmRunner::run,
             "Launch CUTLASS grouped GEMM kernel. setup_meta() must be called first.");

    py::class_<disagmoe::CutlassGemmRunnerFP8, std::shared_ptr<disagmoe::CutlassGemmRunnerFP8>>(m, "CutlassGemmRunnerFP8")
        .def(py::init<torch::Tensor, torch::Tensor, int64_t>(),
             py::arg("fp8_weight"), py::arg("weight_scale"), py::arg("max_tokens"))
        .def("setup_meta", &disagmoe::CutlassGemmRunnerFP8::setup_meta,
             py::arg("a"), py::arg("c"), py::arg("batch_sizes"))
        .def("run", &disagmoe::CutlassGemmRunnerFP8::run);
}
