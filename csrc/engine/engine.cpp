#include "muhelper.h"
#include "logging.h"
#include "utils.hpp"
#include "engine.h"
#include "comm.h"
#include "distributed.hpp"
#include "dispatcher.h"
#include "pool.h"
#include "scheduler.h"

#include <chrono>
#include <thread>
#include <vector>
#include <ctime>
#include <map>

std::tuple<std::vector<Channel_t>, std::vector<Channel_t>> init_all_channels(
    int world_size,
    int local_id,
    bool is_attn,
    // Inter-group Channels
    std::string nccl_comm_id_low_to_high,
    std::string nccl_comm_id_high_to_low,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    int local_attn_dp_rank
) {

    ncclComm_t comm_low_to_high;
    ncclComm_t comm_high_to_low;
    ncclUniqueId nccl_unique_id_low_to_high = string_to_nccl_unique_id(nccl_comm_id_low_to_high);
    ncclUniqueId nccl_unique_id_high_to_low = string_to_nccl_unique_id(nccl_comm_id_high_to_low);
    NCCLCHECK(ncclCommInitRank(&comm_low_to_high, world_size, nccl_unique_id_low_to_high, local_id));
    NCCLCHECK(ncclCommInitRank(&comm_high_to_low, world_size, nccl_unique_id_high_to_low, local_id));
    DMOE_LOG(INFO) << "NCCL comm initialized with world size " << world_size << " and rank " << local_id << LEND;

    auto n_in = in_device_ids.size();
    auto n_out = out_device_ids.size();

    std::vector<Channel_t> in_channels;
    std::vector<Channel_t> out_channels;
    Channel_t local_channel = nullptr;

    // inbound channels
    for (size_t i = 0; i < n_in; i ++) {
        auto peer_id = in_device_ids[i];
        Channel_t channel{};
        if (peer_id == local_id) {
            channel = create_local_channel(local_id);
            local_channel = channel;
        } else if (local_id < peer_id) {
            channel = create_nccl_channel(local_id, peer_id, comm_high_to_low);
        } else {
            channel = create_nccl_channel(local_id, peer_id, comm_low_to_high);
        }
        in_channels.push_back(channel);
    }

    // DMOE_LOG(DEBUG) << local_id << " " << "in channel initialized" << LEND;

    // outbound channels
    for (size_t i = 0; i < n_out; i ++) {
        auto peer_id = out_device_ids[i];
        Channel_t channel{};
        if (peer_id == local_id) {
            channel = local_channel;
        } else if (local_id < peer_id) {
            channel = create_nccl_channel(local_id, peer_id, comm_low_to_high);
        } else {
            channel = create_nccl_channel(local_id, peer_id, comm_high_to_low);
        }
        out_channels.push_back(channel);
    }
    // DMOE_LOG(INFO) << local_id << " " << "all channels initialized" << LEND;
    return std::make_tuple(in_channels, out_channels);
}

std::tuple<mu_pool_t, scheduler_t, mu_dispatcher_t> init_disaggregated_engine(
    int world_size,
    int local_id, 
    int local_attn_dp_rank,
    int top_k,
    bool has_attn,
    bool has_expert,
    bool expert_wise_schedule,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    std::string nccl_comm_id_low_to_high,
    std::string nccl_comm_id_high_to_low,
    const std::vector<ChannelInfo> &out_channel_infos
) {
    ASSERT ((has_attn ^ has_expert) == true);

    auto [in_channels, out_channels] = init_all_channels(
        world_size, local_id, has_attn, 
        nccl_comm_id_low_to_high, nccl_comm_id_high_to_low, 
        in_device_ids, out_device_ids, 
        local_attn_dp_rank
    );

    mu_dispatcher_t dispatcher{};
    scheduler_t scheduler{};
    mu_pool_t pool{};

    if (has_attn) {
        auto attn_dispatcher = std::make_shared<MuAttnDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
        dispatcher = std::static_pointer_cast<MuDispatcher>(attn_dispatcher);
        if (top_k == 1) {
            auto attn_pool = std::make_shared<MuAttentionPool>(layer_ids, local_id, in_channels);
            scheduler = std::make_shared<Scheduler>(attn_pool, mu_expert_pool_t{}, "mbfs");
            pool = std::static_pointer_cast<MuPool>(attn_pool);
        } else {
            auto attn_pool = std::make_shared<MuAttentionTopKPool>(layer_ids, local_id, in_channels, top_k);
            scheduler = std::make_shared<Scheduler>(attn_pool, mu_expert_pool_t{}, "mbfs");
            pool = std::static_pointer_cast<MuPool>(attn_pool);
        }
    } else if (has_expert) {
        auto expert_dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
        dispatcher = std::static_pointer_cast<MuDispatcher>(expert_dispatcher);
        LayerSchedulePolicy policy = LayerSchedulePolicy::GROUP;
        int num_groups = 1;
        if (expert_wise_schedule) {
            throw std::runtime_error("Expert wise schedule is not supported yet");
            policy = LayerSchedulePolicy::GROUP;
            num_groups = cfg.n_exp_per_rank;
            // DMOE_LOG(INFO) << local_id << " expert wise schedule, #experts per EP rank: " << num_groups << LEND;
        }
        auto expert_pool = std::make_shared<MuExpertPool>(layer_ids, local_id, in_channels, num_groups);
        pool = std::static_pointer_cast<MuPool>(expert_pool);
        scheduler = std::make_shared<Scheduler>(mu_attn_pool_t{}, expert_pool, "mbfs");
    }

    return std::make_tuple(pool, scheduler, dispatcher);
}

std::tuple<mu_pool_t, scheduler_t, mu_dispatcher_t> init_unified_engine(
    int world_size,
    int local_id,
    int global_rank,
    int top_k,
    bool has_attn,
    bool has_expert,
    bool expert_wise_schedule,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    std::string nccl_comm_id_low_to_high,
    std::string nccl_comm_id_high_to_low,
    const std::vector<ChannelInfo> &out_channel_infos
) {
    // TODO: support expert wise schedule
    int num_groups = 1;
    int num_layers = layer_ids.size();

    auto [in_channels, out_channels] = init_all_channels(
        world_size, local_id, true, 
        nccl_comm_id_low_to_high, nccl_comm_id_high_to_low, 
        in_device_ids, out_device_ids, 
        global_rank
    );

    auto unified_dispatcher = std::make_shared<UnifiedDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
    auto unified_pool = std::make_shared<UnifiedPool>(layer_ids, local_id, in_channels, num_groups, top_k);
    auto scheduler = std::make_shared<Scheduler>(unified_pool);

    auto casted_pool = std::static_pointer_cast<MuPool>(unified_pool);
    auto casted_dispatcher = std::static_pointer_cast<MuDispatcher>(unified_dispatcher);

    return std::make_tuple(casted_pool, scheduler, casted_dispatcher);
}

void set_hosts(int local_id, const std::map<int, std::string>& device_id_2_ip) {
    set_hosts_internal(local_id, device_id_2_ip);
}

void start_engine(scheduler_t scheduler, mu_dispatcher_t dispatcher) {
    if (scheduler.get() != nullptr)
        scheduler->start();
    if (dispatcher.get() != nullptr)
        dispatcher->start();
}
