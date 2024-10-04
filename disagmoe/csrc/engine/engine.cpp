#include "muhelper.h"
#include "engine.h"
#include "comm.h"

#include <thread>
#include <vector>
#include <map>

std::pair<Scheduler_t, MuDispatcher_t> init_engine(
    int local_id, 
    bool is_attn,
    const std::vector<int> &layer_ids,
    const std::vector<int> &in_device_ids,
    const std::vector<int> &out_device_ids,
    const std::vector<ChannelInfo> &out_channel_infos,
    const std::map<int, uintptr_t> &nccl_ids) {
    int n_in = in_device_ids.size();
    int n_out = out_device_ids.size();

    // init channels
    std::map<int, Channel_t> channels;
    std::vector<Channel_t> in_channels(n_in);
    std::vector<Channel_t> out_channels(n_out);

    for (auto i: in_device_ids)
        channels[i] = nullptr;
    for (auto i: out_device_ids)
        channels[i] = nullptr;
    std::vector<std::thread> threads;
    for (auto &[peer_id, channel]: channels) {
        channel = create_channel(local_id, peer_id, (void*) nccl_ids.at(peer_id));
        threads.push_back(std::thread([&] { 
            channel->instantiate(); 
        }));
    }
    for (auto &t: threads)
        t.join();

    for (int i = 0; i < n_in; i ++)
        in_channels[i] = channels[ in_device_ids[i] ];
    for (int i = 0; i < n_out; i ++)    
        out_channels[i] = channels[ out_device_ids[i] ];
    
    // init dispatcher & pool
    MuPool_t pool = std::make_shared<MuPool>(layer_ids, local_id, in_channels, is_attn);
    MuDispatcher_t dispatcher;
    if (is_attn)
        dispatcher = std::make_shared<MuAttnDispatcher>(layer_ids, local_id, out_channels);
    else
        dispatcher = std::make_shared<MuExpertDispatcher>(layer_ids, local_id, out_channels, out_channel_infos);
    
    // init scheduler
    // TODO(hogura|20241003): add scheduler init config here
    Scheduler_t scheduler = std::make_shared<LargestScheduler>(pool, layer_ids);

    return std::make_pair(scheduler, dispatcher);
}

void start_engine(Scheduler_t scheduler, MuDispatcher_t dispatcher) {
    scheduler->start();
    dispatcher->start();
}