#pragma once

#include <map>

#include "datatypes.hpp"
#include "scheduler.h"
#include "comm.h"

static std::pair<Scheduler_t, MuDispatcher_t> init_engine(
    int local_id, 
    bool is_attn,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, uintptr_t> &nccl_ids
);

static void start_engine(Scheduler_t scheduler, MuDispatcher_t dispatcher);