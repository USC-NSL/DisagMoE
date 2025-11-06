#pragma once

#ifndef DISPATCHER_H_
#define DISPATCHER_H_

#include "muhelper.h"
#include "comm.h"
#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"

class UnifiedDispatcher: public MuDispatcher {

private:
    std::vector<int> expert_to_rank;
    std::vector<int> rank_to_channel;

    inline int _attn_get_channel_id(int dp_rank);

    inline int _expert_get_channel_id(int expert_id);

    void _send_to_expert_once(TokenBatch batch);

    void _send_to_attn_once(TokenBatch batch);

    void _send_once(TokenBatch batch) override;

public:

    UnifiedDispatcher(
        std::vector<int> layer_ids, 
        int device_id, 
        ParallelConfig cfg,
        std::vector<Channel_t> channels={},
        std::vector<ChannelInfo> channel_infos={}
    );

};

#endif