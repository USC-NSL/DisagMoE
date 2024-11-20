#pragma once

#include "muhelper.h"
#include "datatypes.hpp"

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
    std::set<int> eos_seqs; // sequences that have reached EOS
    std::set<int> finished_seqs; // sequences that have reached EOS and ended another round of inference
    std::map<int, SloStat> slo_stats;
    std::map<int, int> output_lens;

    int max_output_len;

    std::mutex result_lock;

    void run() override;

    int _get_attn_channel(int req_id, int layer_id) override;

public:
    Sampler(int device_id, 
            int max_output_len,
            std::vector<Channel_t> in_channels, 
            std::vector<Channel_t> out_channels,
            std::vector<ChannelInfo> out_channel_infos);

    void process_batch(torch::Tensor data, metadata_t meta);

    int sample(uintptr_t buf, metadata_t meta);

    bool check_finished(int token, int req_id);

    void start();

    std::vector<SloStat> fetch_finished_slo_stats();

    std::map<int, SloStat> wait_slo_stats(int n_request);
};

class Tokenizer: public MuExpertDispatcher {
protected:

public:
    Tokenizer(int device_id, 
              std::vector<Channel_t> channels, 
              std::vector<ChannelInfo> out_channel_infos);

    void put_request(int req_id, torch::Tensor tensor);

    void start();
};

typedef std::shared_ptr<Sampler> Sampler_t;
typedef std::shared_ptr<Tokenizer> Tokenizer_t;