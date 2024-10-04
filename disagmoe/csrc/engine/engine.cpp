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

std::pair<Scheduler_t, MuDispatcher_t> init_engine(
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
    for (auto i: out_device_ids)
        channels[i] = nullptr;
    std::vector<std::thread> threads;
    for (auto &[peer_id, channel]: channels) {
        channel = create_channel(local_id, peer_id, 
            convert_to_nccl_uid((char*) nccl_ids.at(peer_id).c_str())
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
    MuPool_t pool = std::make_shared<MuPool>(layer_ids, local_id, in_channels, is_attn);
    MuDispatcher_t dispatcher;
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
    Scheduler_t scheduler = std::make_shared<LargestScheduler>(pool, layer_ids);

    LOG(DEBUG) << local_id << " inited scheduler" << LEND;

    return std::make_pair(scheduler, dispatcher);
}

void start_engine(Scheduler_t scheduler, MuDispatcher_t dispatcher) {
    scheduler->start();
    dispatcher->start();
}