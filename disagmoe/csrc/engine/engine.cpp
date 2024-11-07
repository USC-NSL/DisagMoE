#include "muhelper.h"
#include "logging.h"
#include "utils.hpp"
#include "engine.h"
#include "comm.h"

#include <chrono>
#include <thread>
#include <vector>
#include <ctime>
#include <map>

void init_channels(
    int local_id,
    bool is_attn,
    // P2P Channels
    std::vector<Channel_t> &in_channels,
    std::vector<Channel_t> &out_channels,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    std::map<int, std::pair<std::string, std::string>> &nccl_ids,
    // Group Channels
    Channel_t &tensor_channel,
    const std::vector<int> &tensor_group_device_ids,
    const std::string &tensor_group_nccl_id,
    Channel_t &meta_channel,
    const std::vector<int> &meta_group_device_ids,
    const std::string &meta_group_nccl_id
) {

    LOG(DEBUG) << local_id << " " << "init channels" << LEND;

    int n_in = in_device_ids.size();
    int n_out = out_device_ids.size();

    in_channels.resize(n_in);
    out_channels.resize(n_out);

    #define LocalChannel std::pair<int, bool>
    #define m_id first
    #define m_is_in_channel second
    #define D_IN_CHANNEL 0
    #define D_OUT_CHANNEL 1
    std::map<LocalChannel, Channel_t> channels;

    for (auto i: in_device_ids)
        channels[LocalChannel(i, D_IN_CHANNEL)] = nullptr;
    for (auto i: out_device_ids) {
        ASSERT(!is_embedding_node(i) || channels.find(LocalChannel(i, D_OUT_CHANNEL)) == channels.end());
        channels[LocalChannel(i, D_OUT_CHANNEL)] = nullptr;
    }
    std::vector<std::thread> threads;
    // Init P2P channels
    for (auto &[local_channel, channel]: channels) {
        int peer_id = local_channel.m_id;
        if (!is_embedding_node(peer_id)) {
            auto nccl_id = local_channel.m_is_in_channel ? nccl_ids.at(peer_id).first : nccl_ids.at(peer_id).second;
            channel = create_channel(local_id, peer_id, 
                convert_to_nccl_uid((char*) nccl_id.c_str())
            );
        }
        else
            channel = create_zmq_channel(
                local_id, peer_id, 
                // tokenizer -> (attn/expert); expert -> sampler; sampler -> attn
                /*is_sender=*/ is_tokenizer(peer_id) ? false : !is_attn
            );
        LOG(DEBUG) << local_id << " created a thread" << LEND;
        threads.emplace_back(std::thread(
            [&](Channel_t channel) { 
                LOG(DEBUG) << local_id << " running channel threads @" << channel << LEND;
                channel->instantiate(); 
            }, 
            channel
        ));
    }
    // Init Group channels
    if (tensor_group_device_ids.size() > 1) {
        tensor_channel = create_nccl_group_channel(
            local_id, tensor_group_device_ids, 
            convert_to_nccl_uid((char*) tensor_group_nccl_id.c_str())
        );
        threads.emplace_back(std::thread(
            [&](Channel_t channel) { 
                channel->instantiate(); 
            }, tensor_channel
        ));
    }

    LOG(DEBUG) << local_id << " " << "joining channel threads" << LEND;
    for (auto &t: threads)
        t.join();

    for (int i = 0; i < n_in; i ++)
        in_channels[i] = channels[ LocalChannel(in_device_ids[i], D_IN_CHANNEL) ];
    for (int i = 0; i < n_out; i ++)    
        out_channels[i] = channels[ LocalChannel(out_device_ids[i], D_OUT_CHANNEL) ];
}

std::tuple<scheduler_t, attn_scheduler_t, mu_dispatcher_t> init_engine(
    int local_id, 
    bool is_attn,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    std::map<int, std::pair<std::string, std::string>> &nccl_ids,
    ParallelConfig cfg,
    const std::vector<int> &tensor_group_device_ids,
    const std::string &tensor_group_nccl_id,
    const std::vector<int> &meta_group_device_ids,
    const std::string &meta_group_nccl_id) {

    std::vector<Channel_t> in_channels, out_channels;
    Channel_t tensor_channel, meta_channel;
    init_channels(local_id, is_attn, in_channels, out_channels, 
        in_device_ids, out_device_ids, out_channel_infos, nccl_ids,
        tensor_channel, tensor_group_device_ids, tensor_group_nccl_id,
        meta_channel, meta_group_device_ids, meta_group_nccl_id);
    
    // init dispatcher
    LOG(DEBUG) << local_id << " " << "init dispatcher" << LEND;
    mu_dispatcher_t dispatcher;
    if (is_attn) {
        dispatcher = std::make_shared<MuAttnDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
    }
    else {
        dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
    }
    
    // init scheduler
    LOG(DEBUG) << local_id << " init scheduler" << LEND;
    scheduler_t scheduler;
    attn_scheduler_t attn_scheduler;

    if (is_attn) {
        mu_attn_pool_t pool = std::make_shared<MuAttentionPool>(layer_ids, local_id, in_channels);
        if (cfg.tp == 1) {
            attn_scheduler = AttentionScheduler::build(pool, layer_ids, "largest");
        } else {
            if (local_id == tensor_group_device_ids[0]) {
                // is driver, or saying `root` in the tp group
                attn_scheduler = std::make_shared<AttentionDriverScheduler>(pool, layer_ids, tensor_channel);
            } else {
                attn_scheduler = std::make_shared<AttentionWorkerScheduler>(pool, layer_ids, tensor_channel);
            }
        }
    }
    else {
        mu_pool_t pool = std::make_shared<MuPool>(layer_ids, local_id, in_channels, is_attn);
        scheduler = Scheduler::build(pool, layer_ids, "largest");
    }

    return std::make_tuple(scheduler, attn_scheduler, dispatcher);
}

void start_engine(scheduler_t scheduler, attn_scheduler_t attn_scheduler, mu_dispatcher_t dispatcher) {
    if (scheduler.get() != nullptr)
        scheduler->start();
    if (attn_scheduler.get() != nullptr)
        attn_scheduler->start();
    dispatcher->start();
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
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos
) {
    std::vector<Channel_t> in_channels;
    std::vector<Channel_t> out_channels;
    std::vector<std::thread> threads;

    for (int i: in_device_ids)
        in_channels.push_back(std::make_shared<ZmqChannel>(local, i, false));
    for (int i: out_device_ids)
        out_channels.push_back(std::make_shared<ZmqChannel>(local, i, true));
    INSTANTIATE_CHANNELS(threads, in_channels);
    INSTANTIATE_CHANNELS(threads, out_channels);

    for (auto &t: threads)
        t.join();

    Sampler_t sampler = std::make_shared<Sampler>(
        local, in_channels, out_channels, out_channel_infos
    );
    return sampler;
}

Tokenizer_t init_tokenizer(
    int local,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos
) {
    std::vector<Channel_t> out_channels;
    std::vector<std::thread> threads;

    for (int i: out_device_ids)
        out_channels.push_back(std::make_shared<ZmqChannel>(local, i, true));
    INSTANTIATE_CHANNELS(threads, out_channels);
    
    for (auto &t: threads)
        t.join();

    Tokenizer_t tokenizer = std::make_shared<Tokenizer>(
        local, out_channels, out_channel_infos
    );
    return tokenizer;
}