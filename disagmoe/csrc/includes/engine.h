#pragma once

#include <map>

#include "datatypes.hpp"
#include "scheduler.h"
#include "comm.h"

std::pair<Scheduler_t, MuDispatcher_t> init_engine(
    int local_id, 
    bool is_attn,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    // const std::vector<std::shared_ptr<ChannelInfo>> &out_channel_infos,
    const std::vector<ChannelInfo> &out_channel_infos,
    std::map<int, std::string> &nccl_ids
);

void start_engine(Scheduler_t scheduler, MuDispatcher_t dispatcher);