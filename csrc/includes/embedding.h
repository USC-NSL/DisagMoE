#pragma once

#include "muhelper.h"

#include <set>
#include <memory>

class Sampler: public MuExpertDispatcher {
protected:
    // channels info
    std::vector<Channel_t> peer_channels, out_channels;

    // zmq info
    zmq::context_t ctx;
    zmq::socket_t recv_mq;

    std::vector<zmq::context_t> send_ctxs;
    std::vector<zmq::socket_t> send_mqs;

    // batch processing info
    std::set<int> finished_seqs;
    std::map<int, SloStat> slo_stats;
    std::map<int, int> output_lens;

    std::mutex result_lock;

    void run() override;

    int _get_attn_channel(int req_id, int layer_id) override;

public:
    Sampler(int device_id, 
            std::vector<Channel_t> in_channels, 
            std::vector<Channel_t> out_channels,
            std::vector<ChannelInfo> out_channel_infos);

    void process_batch(uintptr_t data, metadata_t meta);

    int sample(uintptr_t buf, metadata_t meta);

    bool check_finished(int token, int req_id);

    void start();

    std::map<int, SloStat> get_slo_stats(int n_request);
};

class Tokenizer: public MuExpertDispatcher {
protected:
    int req_count;

public:
    Tokenizer(int device_id, 
              std::vector<Channel_t> channels, 
              std::vector<ChannelInfo> out_channel_infos);

    void put_request(uintptr_t buf, std::vector<size_t> shape);

    void start();
};

typedef std::shared_ptr<Sampler> Sampler_t;
typedef std::shared_ptr<Tokenizer> Tokenizer_t;