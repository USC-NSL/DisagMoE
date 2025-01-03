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

void init_tensor_channels(
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
    const std::map<int, vector<int>> &out_group_device_ids,
    const std::map<int, string> &out_group_nccl_ids
) {
    /*
        !NOTE(hogura|20241110): deprecated function
    */
    ASSERT(!is_embedding_node(local_id));
    DMOE_LOG(DEBUG) << local_id << " " << "init channels" << LEND;

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
            if (out_group_device_ids.find(peer_id) != out_group_device_ids.end()
                && out_group_device_ids.at(peer_id).size() > 1) {
                // is a group channel
                ASSERT(out_group_device_ids.at(peer_id)[0] == local_id);
                auto nccl_id = out_group_nccl_ids.at(peer_id);
                channel = create_nccl_group_channel(
                    local_id, out_group_device_ids.at(peer_id), 
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
            } else {
                auto nccl_id = local_channel.m_is_in_channel ? nccl_ids.at(peer_id).first : nccl_ids.at(peer_id).second;
                channel = create_channel(local_id, peer_id, 
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
            }
        }
        else
            channel = create_zmq_channel(
                local_id, peer_id, 
                // tokenizer -> (attn/expert); expert -> sampler; sampler -> attn
                /*is_sender=*/ is_tokenizer(peer_id) ? false : !is_attn
            );
        DMOE_LOG(DEBUG) << local_id << " created a thread" << LEND;
        threads.emplace_back(std::thread(
            [&](Channel_t channel) { 
                DMOE_LOG(DEBUG) << local_id << " running channel threads @" << channel << LEND;
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
                DMOE_LOG(DEBUG) << local_id << " running channel threads @" << channel << LEND;
                channel->instantiate(); 
            }, tensor_channel
        ));
    }

    DMOE_LOG(DEBUG) << local_id << " " << "joining channel threads" << LEND;
    for (auto &t: threads)
        t.join();

    for (int i = 0; i < n_in; i ++)
        in_channels[i] = channels[ LocalChannel(in_device_ids[i], D_IN_CHANNEL) ];
    for (int i = 0; i < n_out; i ++)    
        out_channels[i] = channels[ LocalChannel(out_device_ids[i], D_OUT_CHANNEL) ];
}

void init_all_channels(
    int local_id,
    bool is_attn,
    // Inter-group Channels
    std::vector<Channel_t> &in_channels,
    const std::vector<int> &in_device_ids,
    const std::map<int, std::string> &in_nccl_ids,
    std::vector<Channel_t> &out_channels,
    const std::vector<int> &out_device_ids,
    const std::map<int, std::vector<int>> &out_device_group_ids,
    const std::map<int, std::string> &out_nccl_ids,
    // Intra-group Channels
    Channel_t &intra_group_channel_1, Channel_t &intra_group_channel_2, Channel_t &intra_group_channel_3,
    const std::vector<int> &device_group_ids,
    const std::tuple<std::string, std::string, std::string> &group_nccl_id,
    std::vector<bool> &is_group_channels
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
    std::vector<std::thread> threads;
    auto n_in = in_device_ids.size();
    auto n_out = out_device_ids.size();
    in_channels.resize(n_in);
    out_channels.resize(n_out);
    is_group_channels.resize(n_out);

    #define INST(channel) {                                                                 \
        threads.emplace_back(std::thread(                                                   \
            [&](Channel_t channel) {                                                        \
                DMOE_LOG(DEBUG) << local_id << " running channel threads @" << channel << LEND;  \
                channel->instantiate();                                                     \
                DMOE_LOG(DEBUG) << local_id << " channel @" << channel << " inited" << LEND;     \
            },                                                                              \
            channel                                                                         \
        ));                                                                                 \
    }
    
    for (size_t i = 0; i < n_in; i ++) {
        auto peer_id = in_device_ids[i];
        auto &channel = in_channels[i];
        if (is_embedding_node(peer_id)) {
            channel = create_zmq_channel(local_id, peer_id, /*is_sender=*/ false);
        } else {
            if (device_group_ids.size() > 1) {
                // is a group channel
                // construct the inter-group as [peer_id, device_group_ids]
                std::vector<int> device_ids{};
                device_ids.push_back(peer_id);
                for (auto i: device_group_ids)
                    device_ids.push_back(i);
                auto nccl_id = in_nccl_ids.at(peer_id);
                DMOE_LOG(WARNING) << local_id << " launching nccl group channel " << device_ids.size() << " " << nccl_id << LEND;
                channel = create_nccl_group_channel(
                    local_id, device_ids,
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
            } else {
                auto nccl_id = in_nccl_ids.at(peer_id);
                channel = create_channel(local_id, peer_id, 
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
            }
        }
        INST(channel);
    }

    for (size_t i = 0; i < n_out; i ++) {
        auto peer_id = out_device_ids[i];
        auto &channel = out_channels[i];
        if (is_embedding_node(peer_id)) {
            channel = create_zmq_channel(local_id, peer_id, /*is_sender=*/ true);
        } else {
            if (out_device_group_ids.find(peer_id) != out_device_group_ids.end() 
                && out_device_group_ids.at(peer_id).size() > 2) {
                // size > 2 since [local_id, driver, workers...]
                // is a group channel
                // use out_device_group_ids as intra group
                ASSERT(out_device_group_ids.at(peer_id)[0] == local_id);
                auto nccl_id = out_nccl_ids.at(peer_id);
                DMOE_LOG(WARNING) << local_id << " launching nccl group channel " << out_device_group_ids.at(peer_id).size() << " " << nccl_id << LEND;
                channel = create_nccl_group_channel(
                    local_id, out_device_group_ids.at(peer_id),
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
                is_group_channels[i] = true;
            } else {
                auto nccl_id = out_nccl_ids.at(peer_id);
                channel = create_channel(local_id, peer_id, 
                    convert_to_nccl_uid((char*) nccl_id.c_str())
                );
            }
        }
        INST(channel);
    }

    if (device_group_ids.size() > 1) {
        intra_group_channel_1 = create_nccl_group_channel(
            local_id, device_group_ids, 
            convert_to_nccl_uid((char*) std::get<0>(group_nccl_id).c_str())
        );
        intra_group_channel_2 = create_nccl_group_channel(
            local_id, device_group_ids, 
            convert_to_nccl_uid((char*) std::get<1>(group_nccl_id).c_str())
        );
        intra_group_channel_3 = create_nccl_group_channel(
            local_id, device_group_ids, 
            convert_to_nccl_uid((char*) std::get<2>(group_nccl_id).c_str())
        );
        INST(intra_group_channel_1);
        INST(intra_group_channel_2);
        INST(intra_group_channel_3);
    }

    for (auto &t: threads)
        t.join();
}

std::tuple<scheduler_t, attn_scheduler_t, mu_dispatcher_t> init_engine(
    int local_id, 
    int top_k,
    bool is_attn,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    ParallelConfig cfg,
    const std::map<int, std::string> &in_nccl_ids,
    const std::map<int, std::vector<int>> &out_device_group_ids,
    const std::map<int, std::string> &out_nccl_ids,
    const std::vector<int> device_group_ids,
    const std::tuple<std::string, std::string, std::string> &group_nccl_id) {

    std::vector<Channel_t> in_channels, out_channels;
    std::vector<bool> is_group_channels;
    Channel_t intra_group_channel_1 = nullptr, intra_group_channel_2 = nullptr, intra_group_channel_3 = nullptr;
    // init_tensor_channels(local_id, is_attn, in_channels, out_channels, 
    //     in_device_ids, out_device_ids, out_channel_infos, nccl_ids,
    //     tensor_channel, tensor_group_device_ids, tensor_group_nccl_id,
    //     out_group_device_ids, out_group_nccl_ids);

    init_all_channels(local_id, is_attn, 
        in_channels, in_device_ids, in_nccl_ids,
        out_channels, out_device_ids, out_device_group_ids, out_nccl_ids,
        intra_group_channel_1, intra_group_channel_2, intra_group_channel_3,
        device_group_ids, group_nccl_id, 
        is_group_channels);
    
    // init dispatcher
    DMOE_LOG(DEBUG) << local_id << " " << "init dispatcher" << LEND;
    mu_dispatcher_t dispatcher;
    if (is_attn) {
        dispatcher = std::make_shared<MuAttnDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos);
    }
    else {
        dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, cfg, out_channels, out_channel_infos, is_group_channels);
    }
    
    // init scheduler
    DMOE_LOG(DEBUG) << local_id << " init scheduler" << LEND;
    scheduler_t scheduler;
    attn_scheduler_t attn_scheduler;

    if (is_attn) {
        mu_attn_pool_t pool;
        if (top_k == 1) {
            pool = std::make_shared<MuAttentionPool>(layer_ids, local_id, in_channels, device_group_ids, intra_group_channel_1);
        } else {
            pool = std::make_shared<MuAttentionTopKPool>(layer_ids, local_id, in_channels, device_group_ids, intra_group_channel_1, top_k);
        }
        if (cfg.tp == 1) {
            attn_scheduler = AttentionScheduler::build(pool, layer_ids, "largest");
        } else {
            if (local_id == device_group_ids[0]) {
                // is driver, or saying `root` in the tp group
                attn_scheduler = std::make_shared<AttentionDriverScheduler>(pool, layer_ids, intra_group_channel_2, intra_group_channel_3);
            } else {
                attn_scheduler = std::make_shared<AttentionWorkerScheduler>(pool, layer_ids, intra_group_channel_2, intra_group_channel_3);
            }
        }
    } else {
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
    int max_output_len,
    int top_k,
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

    Sampler_t sampler;
    if (top_k == 1) {
        sampler = std::make_shared<Sampler>(
            local, max_output_len, in_channels, out_channels, out_channel_infos
        );
    } else {
        sampler = std::make_shared<TopKSampler>(
            local, max_output_len, top_k, in_channels, out_channels, out_channel_infos
        );
    }
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

void set_hosts(int local_id, const std::map<int, std::string>& device_id_2_ip) {
    set_hosts_internal(local_id, device_id_2_ip);
}