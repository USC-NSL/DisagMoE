#include "muhelper.h"
#include "logging.h"
#include "utils.hpp"
#include "engine.h"
#include "comm.h"
#include "distributed.hpp"

#include <chrono>
#include <thread>
#include <vector>
#include <ctime>
#include <map>

void init_all_channels(
    int local_id,
    bool is_attn,
    // Inter-group Channels
    std::vector<Channel_t> &in_channels,
    const std::vector<int> &in_device_ids,
    const std::map<int, std::string> &in_nccl_ids,
    std::vector<Channel_t> &out_channels,
    const std::vector<int> &out_device_ids,
    const std::map<int, std::string> &out_nccl_ids,
    int local_attn_dp_rank,
    std::vector<std::thread>& threads,
    bool skip_embedding = false
) {
    /*
        We have multiple types of channels to be inited.

        TP_SIZE = device_group_ids.size()

        * Normal P2P channel (TP_SIZE = 1)

        * Inter Group P2P channel (TP_SIZE > 1)
            * Both for in_device_ids and out_device_ids
        * Intra Group P2P channel (TP_SIZE > 1)
            * Only for device_group_ids
            
    */
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
            channel = create_zmq_channel(local_id, peer_id, /*is_sender=*/ false, 
                // only attn needs to consider the DP
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
            channel = create_zmq_channel(local_id, peer_id, /*is_sender=*/ true);
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

    // DMOE_LOG(DEBUG) << local_id << " " << "out channel initialized" << LEND;
}

std::tuple<attn_scheduler_t, mu_dispatcher_t, scheduler_t, mu_dispatcher_t> init_engine(
    int local_id, 
    int top_k,
    bool has_attn,
    bool has_expert,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids,
    const std::vector<int> &device_group_ids,
    int local_attn_dp_rank) {

    ASSERT (has_attn ^ has_expert == true);

    std::vector<Channel_t> in_channels, out_channels;
    std::vector<bool> is_group_channels;
    Channel_t intra_group_channel_1 = nullptr;

    std::vector<std::thread> threads;

    init_all_channels(local_id, has_attn,
        in_channels, in_device_ids, in_nccl_ids,
        out_channels, out_device_ids, out_nccl_ids,
        local_attn_dp_rank, threads);

    for (auto &t: threads)
        t.join();

    std::cout << local_id << " finished init all channels" << std::endl;
    
    // init dispatcher
    // DMOE_LOG(DEBUG) << local_id << " " << "init dispatcher" << LEND;
    mu_dispatcher_t attn_dispatcher{}, expert_dispatcher{};
    if (has_attn) {
        attn_dispatcher = std::make_shared<MuAttnDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
    }
    if (has_expert) {
        expert_dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos, is_group_channels);
    }
    
    // init scheduler
    // DMOE_LOG(DEBUG) << local_id << "init scheduler" << LEND;
    scheduler_t expert_scheduler;
    attn_scheduler_t attn_scheduler;

    if (has_attn) {
        mu_attn_pool_t pool;
        if (top_k == 1) {
            pool = std::make_shared<MuAttentionPool>(layer_ids, local_id, in_channels, device_group_ids, intra_group_channel_1);
        } else {
            pool = std::make_shared<MuAttentionTopKPool>(layer_ids, local_id, in_channels, device_group_ids, intra_group_channel_1, top_k);
        }
        attn_scheduler = AttentionScheduler::build(pool, layer_ids, "mbfs");
    } 
    if (has_expert) {
        mu_pool_t pool = std::make_shared<MuPool>(layer_ids, local_id, in_channels);
        expert_scheduler = Scheduler::build(pool, layer_ids, "mbfs");
    }

    return std::make_tuple(attn_scheduler, attn_dispatcher, expert_scheduler, expert_dispatcher);
}


std::tuple<attn_scheduler_t, mu_dispatcher_t, scheduler_t, mu_dispatcher_t> init_engine_colocate(
    int local_id, 
    int top_k,
    bool has_attn,
    bool has_expert,
    ParallelConfig cfg,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::string> &out_nccl_ids,
    const std::map<int, std::string> &in_nccl_ids_ext,
    const std::map<int, std::string> &out_nccl_ids_ext,
    const std::vector<int> &device_group_ids,
    int local_attn_dp_rank) {

    ASSERT (has_attn && has_expert);

    std::vector<Channel_t> in_channels, out_channels;
    std::vector<Channel_t> in_channels_ext, out_channels_ext;
    std::vector<bool> is_group_channels;
    Channel_t intra_group_channel_1 = nullptr;

    std::vector<std::thread> threads;

    init_all_channels(local_id, has_attn,
        in_channels, in_device_ids, in_nccl_ids,
        out_channels, out_device_ids, out_nccl_ids,
        local_attn_dp_rank, threads, true); // skip embedding

    init_all_channels(local_id, has_attn,
        in_channels_ext, in_device_ids, in_nccl_ids_ext,
        out_channels_ext, out_device_ids, out_nccl_ids_ext,
        local_attn_dp_rank, threads);

    for (auto &t: threads)
        t.join();

    std::cout << local_id << " finished init all channels" << std::endl;
    
    // init dispatcher
    // DMOE_LOG(DEBUG) << local_id << " " << "init dispatcher" << LEND;
    mu_dispatcher_t attn_dispatcher{}, expert_dispatcher{};
    attn_dispatcher = std::make_shared<MuAttnDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
    expert_dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, cfg, out_channels_ext, out_channel_infos, is_group_channels);
    
    // init scheduler
    // DMOE_LOG(DEBUG) << local_id << "init scheduler" << LEND;
    scheduler_t expert_scheduler;
    attn_scheduler_t attn_scheduler;

    {
        // init attention scheduler
        mu_attn_pool_t pool;
        if (top_k == 1) {
            pool = std::make_shared<MuAttentionPool>(layer_ids, local_id, in_channels_ext, device_group_ids, intra_group_channel_1);
        } else {
            pool = std::make_shared<MuAttentionTopKPool>(layer_ids, local_id, in_channels_ext, device_group_ids, intra_group_channel_1, top_k);
        }
        attn_scheduler = AttentionScheduler::build(pool, layer_ids, "mbfs");
    }

    {
        // init expert scheduler
        mu_pool_t pool = std::make_shared<MuPool>(layer_ids, local_id, in_channels);
        expert_scheduler = Scheduler::build(pool, layer_ids, "mbfs");
    }

    return std::make_tuple(attn_scheduler, attn_dispatcher, expert_scheduler, expert_dispatcher);
}


void start_engine(attn_scheduler_t attn_scheduler, mu_dispatcher_t attn_dispatcher, scheduler_t expert_scheduler, mu_dispatcher_t expert_dispatcher) {
    if (attn_scheduler.get() != nullptr)
        attn_scheduler->start();
    if (expert_scheduler.get() != nullptr)
        expert_scheduler->start();
    if (attn_dispatcher.get() != nullptr)
        attn_dispatcher->start();
    if (expert_dispatcher.get() != nullptr)
        expert_dispatcher->start();
}

#define INSTANTIATE_CHANNELS(threads, _channels) {  \
    for (auto &chan: _channels)                     \
        threads.push_back(std::thread(              \
            [&](Channel_t c) {                      \
                c->instantiate();                   \
            }, chan                                 \
        ));                                         \
}

Sampler_t init_sampler(
    int local,
    int max_output_len,
    int top_k,
    ParallelConfig cfg,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos
) {
    std::vector<Channel_t> in_channels;
    std::vector<Channel_t> out_channels;
    std::vector<std::thread> threads;

    for (int i: in_device_ids)
        in_channels.push_back(std::make_shared<ZmqChannel>(local, i, false));
    
    ASSERT(out_device_ids.size() == out_channel_infos.size());
    for (int id = 0; id < out_device_ids.size(); id ++) {
        int i = out_device_ids[id];
        int rank = out_channel_infos[id].attn_dp_rank;
        out_channels.push_back(std::make_shared<ZmqChannel>(local, i, true, rank));
    }
    INSTANTIATE_CHANNELS(threads, in_channels);
    INSTANTIATE_CHANNELS(threads, out_channels);

    for (auto &t: threads)
        t.join();

    Sampler_t sampler;
    if (top_k == 1) {
        sampler = std::make_shared<Sampler>(
            local, max_output_len, cfg, in_channels, out_channels, out_channel_infos
        );
    } else {
        sampler = std::make_shared<TopKSampler>(
            local, max_output_len, top_k, cfg, in_channels, out_channels, out_channel_infos
        );
    }
    return sampler;
}

Tokenizer_t init_tokenizer(
    int local,
    ParallelConfig cfg,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos
) {
    std::vector<Channel_t> out_channels;
    std::vector<std::thread> threads;

    ASSERT(out_device_ids.size() == out_channel_infos.size());
    for (int id = 0; id < out_device_ids.size(); id ++) {
        int i = out_device_ids[id];
        int rank = out_channel_infos[id].attn_dp_rank;
        out_channels.push_back(std::make_shared<ZmqChannel>(local, i, true, rank));
    }
    INSTANTIATE_CHANNELS(threads, out_channels);
    
    for (auto &t: threads)
        t.join();

    Tokenizer_t tokenizer = std::make_shared<Tokenizer>(
        local, cfg, out_channels, out_channel_infos
    );
    return tokenizer;
}

void set_hosts(int local_id, const std::map<int, std::string>& device_id_2_ip) {
    set_hosts_internal(local_id, device_id_2_ip);
}