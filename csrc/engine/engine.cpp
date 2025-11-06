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
    int local_id,
    bool is_attn,
    // Inter-group Channels
    const std::vector<int> &in_device_ids,
    const std::map<int, std::string> &in_nccl_ids,
    const std::vector<int> &out_device_ids,
    const std::map<int, std::string> &out_nccl_ids,
    int local_attn_dp_rank,
    bool skip_embedding = false
) {

    std::vector<std::thread> threads;

    auto n_in = in_device_ids.size();
    auto n_out = out_device_ids.size();

    #define INST(channel, msg) {                                                                 \
        threads.emplace_back(std::thread(                                                   \
            [local_id, peer_id](Channel_t channel) {                                                        \
                channel->instantiate();                                                     \
            },                                                                              \
            channel                                                                         \
        ));                                                                                 \
    }

    std::vector<Channel_t> in_channels;
    std::vector<Channel_t> out_channels;
    Channel_t local_channel = nullptr;

    // print all in_device_ids
    std::cout << local_id << " in_device_ids: ";
    for (auto id: in_device_ids)
        std::cout << id << " ";
    std::cout << std::endl;

    for (size_t i = 0; i < n_in; i ++) {
        auto peer_id = in_device_ids[i];
        Channel_t channel{};
        if (is_embedding_node(peer_id)) {
            if (skip_embedding) {
                continue;
            }
            const auto &make_embed = disagmoe::embedding_channel_factory();
            channel = make_embed(local_id, peer_id, /*is_sender=*/ false,
                is_attn ? local_attn_dp_rank : 0);
        } else {
            if (peer_id == local_id) {
                channel = create_local_channel(local_id);
                local_channel = channel;
            } else {
                auto nccl_id = in_nccl_ids.at(peer_id);
                channel = create_channel(local_id, peer_id, 
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
            }
        }
        in_channels.push_back(channel);
        INST(channel, std::string("in channel=== ") + std::to_string(local_id) + "<-" + std::to_string(peer_id));
    }

    // DMOE_LOG(DEBUG) << local_id << " " << "in channel initialized" << LEND;

    for (size_t i = 0; i < n_out; i ++) {
        auto peer_id = out_device_ids[i];
        Channel_t channel{};
        if (is_embedding_node(peer_id)) {
            if (skip_embedding) {
                continue;
            }
            const auto &make_embed = disagmoe::embedding_channel_factory();
            channel = make_embed(local_id, peer_id, /*is_sender=*/ true, /*rank=*/ 0);
        } else {
            if (peer_id == local_id) {
                channel = local_channel;
            } else {
                auto nccl_id = out_nccl_ids.at(peer_id);
                channel = create_channel(local_id, peer_id, 
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
            }
        }
        out_channels.push_back(channel);
        INST(channel, std::string("out channel=== ") + std::to_string(local_id) + "->" + std::to_string(peer_id));
    }

    for (auto &t: threads)
        t.join();

    // DMOE_LOG(INFO) << local_id << " " << "all channels initialized" << LEND;
    return std::make_tuple(in_channels, out_channels);
}

std::tuple<mu_pool_t, scheduler_t, mu_dispatcher_t> init_disaggregated_engine(
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
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids
) {
    ASSERT ((has_attn ^ has_expert) == true);

    auto [in_channels, out_channels] = init_all_channels(local_id, has_attn, in_device_ids, in_nccl_ids, out_device_ids, out_nccl_ids, local_attn_dp_rank);

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
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids
) {
    // TODO: support expert wise schedule
    int num_groups = 1;
    int num_layers = layer_ids.size();

    auto [in_channels, out_channels] = init_all_channels(local_id, true, in_device_ids, in_nccl_ids, out_device_ids, out_nccl_ids, global_rank);

    auto unified_dispatcher = std::make_shared<UnifiedDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
    auto unified_pool = std::make_shared<UnifiedPool>(layer_ids, local_id, in_channels, num_groups, top_k);
    auto scheduler = std::make_shared<Scheduler>(unified_pool);

    auto casted_pool = std::static_pointer_cast<MuPool>(unified_pool);
    auto casted_dispatcher = std::static_pointer_cast<MuDispatcher>(unified_dispatcher);

    return std::make_tuple(casted_pool, scheduler, casted_dispatcher);
}

#define INSTANTIATE_CHANNELS(threads, _channels) {  \
    for (auto &chan: _channels)                     \
        threads.push_back(std::thread(              \
            [&](Channel_t c) {                      \
                c->instantiate();                   \
            }, chan                                 \
        ));                                         \
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
