#pragma once

#include <map>

#include "datatypes.hpp"
#include "scheduler.h"
#include "embedding.h"
#include "comm.h"

std::pair<Scheduler_t, MuDispatcher_t> init_engine(
    int local_id, 
    bool is_attn,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    std::map<int, std::string> &nccl_ids
);

void start_engine(Scheduler_t scheduler, MuDispatcher_t dispatcher);

Sampler_t init_sampler(
    int device_id,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos
);

Tokenizer_t init_tokenizer(
    int device_id,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos
);