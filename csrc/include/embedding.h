#pragma once

#include "muhelper.h"
#include "datatypes.hpp"

#include <set>
#include <memory>

struct SamplerStepInfo {
    int num_tokens;
    float time_stamp;
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
    std::set<int> eos_seqs; // sequences that have reached EOS
    std::set<int> finished_seqs; // sequences that have reached EOS and ended another round of inference
    std::map<int, SloStat> slo_stats;
    std::map<int, int> output_lens;
    std::unordered_map<int, int> required_output_lens;

    std::vector<SamplerStepInfo> step_infos; 

    int min_output_len;
    int max_output_len;

    std::mutex result_lock;

    int _active_token_count = 0;

    void run() override;

    int _get_attn_channel(int req_id, int layer_id) override;

public:
    Sampler(int device_id, 
            int min_output_len,
            int max_output_len,
            ParallelConfig cfg,
            std::vector<Channel_t> in_channels, 
            std::vector<Channel_t> out_channels,
            std::vector<ChannelInfo> out_channel_infos);

    virtual int process_batch(torch::Tensor data, metadata_t meta);

    int sample(uintptr_t buf, metadata_t meta);

    bool check_finished(int token, int req_id);

    void start();

    std::vector<SloStat> fetch_finished_slo_stats();

    std::vector<SamplerStepInfo> fetch_step_infos();

    void reset();

    std::map<int, SloStat> wait_slo_stats(int n_request);
};

class TopKSampler: public Sampler {
private:
    TokenTopKPool token_pool;

    int top_k;

public:

    TopKSampler(int device_id, 
                int min_output_len,
                int max_output_len,
                int top_k,
                ParallelConfig cfg,
                std::vector<Channel_t> in_channels, 
                std::vector<Channel_t> out_channels,
                std::vector<ChannelInfo> out_channel_infos);

    int process_batch(torch::Tensor data, metadata_t meta) override;

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