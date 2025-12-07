#pragma once

#include <map>

#include "datatypes.hpp"
#include "pool.h"
#include "dispatcher.h"
#include "scheduler.h"
#include "comm.h"

using std::vector;
using std::string;

std::tuple<mu_pool_t, scheduler_t, mu_dispatcher_t> init_disaggregated_engine(
    int world_size,
    int local_id,
    int local_attn_dp_rank,  // DP rank
    int top_k,
    bool has_attn,
    bool has_expert,
    bool expert_wise_schedule,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    // P2P Channels
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    std::map<int, std::string> inbound_nccl_ids,
    std::map<int, std::string> outbound_nccl_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::string &unified_scheduler_type,
    float defrag_weight_decay,
    int defrag_lookahead_steps,
    int defrag_lookback_steps);

std::tuple<mu_pool_t, scheduler_t, mu_dispatcher_t> init_unified_engine(
    int world_size,
    int local_id,
    int global_rank,  // rank in group
    int top_k,
    bool has_attn,
    bool has_expert,
    bool expert_wise_schedule,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    // P2P Channels
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    std::map<int, std::string> inbound_nccl_ids,
    std::map<int, std::string> outbound_nccl_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::string &unified_scheduler_type,
    float defrag_weight_decay,
    int defrag_lookahead_steps,
    int defrag_lookback_steps);

void start_engine(scheduler_t scheduler, mu_dispatcher_t dispatcher);

void set_hosts(int process_id, const std::map<int, std::string>& device_id_2_ip);
