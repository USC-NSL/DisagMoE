#pragma once

#include <map>

#include "datatypes.hpp"
#include "scheduler.h"
#include "embedding.h"
#include "comm.h"

using std::vector;
using std::string;

std::tuple<scheduler_t, attn_scheduler_t, mu_dispatcher_t> init_engine(
    int local_id, 
    bool is_attn,
    const std::vector<int> &layer_ids,
    // P2P Channels
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    // Parallel config
    ParallelConfig cfg,
    // group channels
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::vector<int>> &out_device_group_ids,
    const std::map<int, std::string> &out_nccl_ids,
    const std::vector<int> device_group_ids,
    const std::tuple<std::string, std::string, std::string> &group_nccl_id,
    // DP rank
    int local_attn_dp_rank
);

void start_engine(scheduler_t scheduler, attn_scheduler_t attn_scheduler, mu_dispatcher_t dispatcher);

Sampler_t init_sampler(
    int device_id,
    int max_output_len,
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