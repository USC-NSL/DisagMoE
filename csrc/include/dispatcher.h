#pragma once

#ifndef DISPATCHER_H_
#define DISPATCHER_H_

#include "muhelper.h"
#include "comm.h"
#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"

#include <queue>

struct PendingRankSend {
    TokenBatch batch;
    cudaEvent_t event;
    int dest_rank;
};

struct RankQueue {
    std::vector<TokenBatch> buffered;
    int in_flight = 0;
    int channel_id = -1;
};

class UnifiedDispatcher: public MuDispatcher {

private:
    std::vector<int> expert_to_rank;
    std::vector<int> rank_to_channel;

    std::queue<PendingRankSend> rank_pending_sends_;
    std::vector<RankQueue> rank_queues_;
    int max_in_flight_per_rank_{2};

    inline int _attn_get_channel_id(int dp_rank);

    inline int _expert_get_channel_id(int expert_id);

    void _send_to_expert_once(TokenBatch batch);

    void _send_to_attn_once(TokenBatch batch);

    void _send_once(TokenBatch batch) override;

    void _enqueue_for_rank(int dest_rank, TokenBatch batch);

    void _try_flush_queues();

    void _clean_rank_pending_sends();

    void _do_rank_send(int dest_rank, const TokenBatch& batch);

    TokenBatch _merge_for_rank(std::vector<TokenBatch>& batches);

    bool _has_buffered_sends() const;

    void run() override;

public:

    UnifiedDispatcher(
        std::vector<int> layer_ids, 
        int device_id, 
        ParallelConfig cfg,
        std::vector<Channel_t> channels={},
        std::vector<ChannelInfo> channel_infos={}
    );

    void set_max_in_flight_per_rank(int val) { max_in_flight_per_rank_ = val; }

    void terminate() override;

};

#endif
