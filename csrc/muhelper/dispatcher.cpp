#include "dispatcher.h"

int UnifiedDispatcher::_attn_get_channel_id(int dp_rank) {
    return this->rank_to_channel[dp_rank];
}

int UnifiedDispatcher::_expert_get_channel_id(int expert_id) {
    int rank = this->expert_to_rank[expert_id];
    return this->rank_to_channel[rank];
}

UnifiedDispatcher::UnifiedDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    ParallelConfig cfg,
    std::vector<Channel_t> channels,
    std::vector<ChannelInfo> channel_infos
): MuDispatcher(layer_ids, device_id, cfg, channels) {
    // a channel must contain an attention dp and experts
    // Use n_total_experts when set (asymmetric placement); otherwise fall
    // back to the legacy uniform formula.
    int num_experts = cfg.n_total_experts > 0 ? cfg.n_total_experts
                                              : cfg.ep * cfg.n_exp_per_rank;
    this->rank_to_channel.resize(cfg.dp, -1);
    this->expert_to_rank.resize(num_experts, -1);

    for (auto &tuple: cfg.expert_ranks) {
        int layer_id = std::get<0>(tuple);
        int exp_id = std::get<1>(tuple);
        int rank = std::get<2>(tuple);
        this->expert_to_rank[exp_id] = rank;
    }

    for (size_t i = 0; i < channels.size(); i ++) {
        int dp_rank = channel_infos[i].attn_dp_rank;
        this->rank_to_channel[dp_rank] = i;
    }
}

void UnifiedDispatcher::_send_once(TokenBatch batch) {
    if (batch.metadata->is_attention()) {
        this->_send_to_expert_once(batch);
    } else if (batch.metadata->is_expert()) {
        this->_send_to_attn_once(batch);
    }
}

void UnifiedDispatcher::_send_to_expert_once(TokenBatch batch) {
    tx_range _{"UnifiedDispatcher::_send_to_expert_once"};
    // DMOE_LOG(INFO) << "attn " << this->device_id << " sending a batch: " << *batch.metadata << LEND;
    std::vector<int> split_sizes{};
    for (int i = 0, j = 1, n = batch.metadata->exp_ids.size(); i < n; i = j) {
        int rank = batch.metadata->exp_ids[i];
        int channel_id = this->_expert_get_channel_id(rank);
        while (j < n && this->_expert_get_channel_id(batch.metadata->exp_ids[j]) == channel_id)
            j ++;
        split_sizes.push_back(j - i);
    }

    auto batches = batch.split_with_sizes(split_sizes);
    for (auto &batch: batches) {
        int channel_id = this->_expert_get_channel_id(batch.metadata->exp_ids[0]);
        this->send_batch_nonblocking(channel_id, batch);
    }
}

void UnifiedDispatcher::_send_to_attn_once(TokenBatch batch) {
    tx_range _{"UnifiedDispatcher::_send_to_attn_once"};
    // DMOE_LOG(INFO) << "expert " << device_id << " sending a batch: " << *batch.metadata << ", n_ele=" << batch.data.numel()  << LEND;
    std::vector<int> split_sizes{};
    for (int i = 0, j = 1, n = batch.metadata->attn_dp_ranks.size(); i < n; i = j) {
        int rank = batch.metadata->attn_dp_ranks[i];
        while (j < n && batch.metadata->attn_dp_ranks[j] == rank)
            j ++;
        split_sizes.push_back(j - i);
    }

    auto batches = batch.split_with_sizes(split_sizes);
    for (auto &batch: batches) {
        int channel_id = this->_attn_get_channel_id(batch.metadata->attn_dp_ranks[0]);
        this->send_batch_nonblocking(channel_id, batch);
    }
}