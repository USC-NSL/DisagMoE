#pragma once

#include <map>

#include "datatypes.hpp"
#include "scheduler.h"
#include "embedding.h"
#include "comm.h"

using std::vector;
using std::string;

std::tuple<attn_scheduler_t, mu_dispatcher_t, scheduler_t, mu_dispatcher_t> init_engine(
    int local_id, 
    int top_k,
    bool has_attn,
    bool has_expert,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    // P2P Channels
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids,
    const std::vector<int> &device_group_ids,
    int local_attn_dp_rank // DP rank
);

std::tuple<attn_scheduler_t, mu_dispatcher_t, scheduler_t, mu_dispatcher_t> init_engine_colocate(
    int local_id, 
    int top_k,
    bool has_attn,
    bool has_expert,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids,
    const std::map<int, std::string> &in_nccl_ids_ext,
    const std::map<int, std::string> &out_nccl_ids_ext,
    const std::vector<int> &device_group_ids,
    int local_attn_dp_rank
);

void start_engine(attn_scheduler_t attn_scheduler, mu_dispatcher_t attn_dispatcher, scheduler_t expert_scheduler, mu_dispatcher_t expert_dispatcher);

Sampler_t init_sampler(
    int device_id,
    int min_output_len,
    int max_output_len,
    int top_k,
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