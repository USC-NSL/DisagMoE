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

std::pair<scheduler_t, mu_dispatcher_t> init_engine(
    int local_id, 
    bool is_attn,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    // const std::vector<std::shared_ptr<ChannelInfo>> &out_channel_infos,
    const std::vector<ChannelInfo> &out_channel_infos,
    std::map<int, std::string> &nccl_ids) {
    int n_in = in_device_ids.size();
    int n_out = out_device_ids.size();

    // init channels
    LOG(DEBUG) << local_id << " " << "init channels" << LEND;
    std::map<int, Channel_t> channels;
    std::vector<Channel_t> in_channels(n_in);
    std::vector<Channel_t> out_channels(n_out);

    for (auto i: in_device_ids)
        channels[i] = nullptr;
    for (auto i: out_device_ids) {
        assert(!is_embedding_node(i) || channels.find(i) == channels.end());
        channels[i] = nullptr;
    }
    std::vector<std::thread> threads;
    for (auto &[peer_id, channel]: channels) {
        if (!is_embedding_node(peer_id))
            channel = create_channel(local_id, peer_id, 
                convert_to_nccl_uid((char*) nccl_ids.at(peer_id).c_str())
            );
        else
            channel = create_zmq_channel(
                local_id, peer_id, 
                // tokenizer -> (attn/expert); expert -> sampler; sampler -> attn
                /*is_sender=*/ is_tokenizer(peer_id) ? false : !is_attn
            );
        LOG(DEBUG) << local_id << " created a thread" << LEND;
        threads.push_back(std::thread(
            [&](Channel_t channel) { 
                LOG(DEBUG) << local_id << " running channel threads @" << channel << LEND;
                channel->instantiate(); 
                // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }, 
            channel
        ));
    }
    LOG(DEBUG) << local_id << " " << "joining channel threads" << LEND;
    for (auto &t: threads)
        t.join();

    for (int i = 0; i < n_in; i ++)
        in_channels[i] = channels[ in_device_ids[i] ];
    for (int i = 0; i < n_out; i ++)    
        out_channels[i] = channels[ out_device_ids[i] ];
    
    // init dispatcher & pool
    LOG(DEBUG) << local_id << " " << "init dispatcher & pool" << LEND;
    mu_pool_t pool = std::make_shared<MuPool>(layer_ids, local_id, in_channels, is_attn);
    mu_dispatcher_t dispatcher;
    if (is_attn) {
        dispatcher = std::make_shared<MuAttnDispatcher>(layer_ids, local_id, out_channels);
    }
    else {
        // std::vector<ChannelInfo> _out_channel_infos;
        // for (auto info: out_channel_infos)
        //     _out_channel_infos.push_back(*info);
        // dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, out_channels, _out_channel_infos);
        dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, out_channels, out_channel_infos);
    }
    
    LOG(DEBUG) << local_id << " init scheduler" << LEND;
    // init scheduler
    // TODO(hogura|20241003): add scheduler init config here
    scheduler_t scheduler = Scheduler::build(pool, layer_ids, "largest");

    LOG(DEBUG) << local_id << " inited scheduler" << LEND;

    return std::make_pair(scheduler, dispatcher);
}

void start_engine(scheduler_t scheduler, mu_dispatcher_t dispatcher) {
    scheduler->start();
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