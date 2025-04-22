#pragma once

#include "muhelper.h"
#include "datatypes.hpp"

#include <set>
#include <memory>

struct SamplerStepInfo {
    int num_tokens;
    long long time_stamp;

    SamplerStepInfo() : num_tokens(0), time_stamp(0) {}
    
    SamplerStepInfo(int num_tokens, long long time_stamp):
        num_tokens(num_tokens), time_stamp(time_stamp) {}
};

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
    std::set<int> finished_seqs; // sequences that have reached EOS and ended another round of inference
    std::map<int, SloStat> slo_stats;

    std::vector<SamplerStepInfo> step_infos; 

    std::mutex result_lock;

    int _active_token_count = 0;

    void run() override;

public:
    Sampler(int device_id, 
            ParallelConfig cfg,
            std::vector<Channel_t> in_channels, 
            std::vector<Channel_t> out_channels,
            std::vector<ChannelInfo> out_channel_infos);

    int process_batch(torch::Tensor data, metadata_t meta);

    void start();

    std::vector<SloStat> fetch_finished_slo_stats();

    std::vector<SamplerStepInfo> fetch_step_infos();

    void reset();

    std::map<int, SloStat> wait_slo_stats(int n_request);
};

class Tokenizer: public MuExpertDispatcher {
protected:

public:
    Tokenizer(int device_id, 
              ParallelConfig cfg,
              std::vector<Channel_t> channels, 
              std::vector<ChannelInfo> out_channel_infos);

    void put_request(int req_id, int init_prefill_len, torch::Tensor tensor, int dp_rank);

    void start();

};

typedef std::shared_ptr<Sampler> Sampler_t;
typedef std::shared_ptr<Tokenizer> Tokenizer_t;