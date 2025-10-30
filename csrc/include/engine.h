#pragma once

#include <map>

#include "datatypes.hpp"
#include "pool.h"
#include "dispatcher.h"
#include "scheduler.h"
#include "embedding.h"
#include "comm.h"

using std::vector;
using std::string;

std::tuple<mu_pool_t, scheduler_t, mu_dispatcher_t> init_disaggregated_engine(
    int local_id, 
    int local_attn_dp_rank, // DP rank
    int top_k,
    bool has_attn,
    bool has_expert,
    bool expert_wise_schedule,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    // P2P Channels
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids
);

std::tuple<mu_pool_t, scheduler_t, mu_dispatcher_t> init_unified_engine(
    int local_id,
    int global_rank, // rank in group
    int top_k,
    bool has_attn,
    bool has_expert,
    bool expert_wise_schedule,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    // P2P Channels
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids
);

void start_engine(scheduler_t scheduler, mu_dispatcher_t dispatcher);

Sampler_t init_sampler(
    int device_id,
    ParallelConfig cfg,
    const vector<int> &in_device_ids,
    const vector<int> &out_device_ids,
    const vector<ChannelInfo> &out_channel_infos
);

Tokenizer_t init_tokenizer(
    int device_id,
    ParallelConfig cfg,
    const vector<int> &out_device_ids,
    const vector<ChannelInfo> &out_channel_infos
);

void set_hosts(int process_id, const std::map<int, std::string>& device_id_2_ip);
