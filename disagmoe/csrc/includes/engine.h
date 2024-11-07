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
    const vector<int> &layer_ids,
    // P2P Channels
    const vector<int> &in_device_ids,
    const vector<int> &out_device_ids,
    const vector<ChannelInfo> &out_channel_infos,
    std::map<int, std::pair<string, string>> &nccl_ids,
    // Parallel Config
    ParallelConfig cfg,
    // Group Channels
    const vector<int> &tensor_group_device_ids,
    const string &tensor_group_nccl_id,
    const vector<int> &meta_group_device_ids,
    const string &meta_group_nccl_id
);

void start_engine(scheduler_t scheduler, attn_scheduler_t attn_scheduler, mu_dispatcher_t dispatcher);

Sampler_t init_sampler(
    int device_id,
    const vector<int> &in_device_ids,
    const vector<int> &out_device_ids,
    const vector<ChannelInfo> &out_channel_infos
);

Tokenizer_t init_tokenizer(
    int device_id,
    const vector<int> &out_device_ids,
    const vector<ChannelInfo> &out_channel_infos
);