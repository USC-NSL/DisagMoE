#include "distributed.hpp"
#include "embedding.h"
#include "constants.h"
#include "logging.h"
#include "utils.hpp"

#include "zmq.hpp"
#include "zmq_addon.hpp"

#include <thread>
#include <ctime>
#include <chrono>

Sampler::Sampler(int device_id, 
                ParallelConfig cfg,
                std::vector<Channel_t> in_channels, 
                std::vector<Channel_t> out_channels,
                std::vector<ChannelInfo> out_channel_infos):
    MuExpertDispatcher(
        /*layer_ids=*/ {0}, 
        device_id, 
        ParallelConfig(1, 1, cfg.dp, 1, {}),
        out_channels, 
        out_channel_infos
) {
    
    ctx = zmq::context_t(in_channels.size());
    recv_mq = zmq::socket_t(ctx, zmq::socket_type::pull);

    // copied from MuPool
    int max_peer_id = 0;
    for (auto c: in_channels)
        max_peer_id = std::max(max_peer_id, c->get_peer_id());
    this->peer_channels = std::vector<Channel_t>(max_peer_id + 1);
    for (size_t i = 0; i < in_channels.size(); i ++) {
        int id = in_channels[i]->get_peer_id();
        ASSERT(this->peer_channels[id].get() == nullptr);
        this->peer_channels[ in_channels[i]->get_peer_id() ] = in_channels[i];
    }
    DMOE_LOG(INFO) << "inited sampler" << LEND;
}

void Sampler::run() {
    this->recv_mq.bind(get_zmq_addr(device_id, true, -1, 0));
    // for (int i = 0; i < this->channels.size(); i ++)
    //     this->peer_mq[i].connect(get_zmq_addr(this->channels[i]->get_peer_id(), true, -1, 1));

    int token_processed = 0;
    int iter = 0;
    long long start_timestamp_ms = t_now_high();

    while (!this->end_flag) {
        // DMOE_LOG(WARNING) << "sampler receiving msg ..." << LEND;
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_result_t result =
            zmq::recv_multipart(this->recv_mq, std::back_inserter(recv_msgs));
        
        // DMOE_LOG(WARNING) << "sampler got msg !!!" << LEND;
        ASSERT(*result == 2);
        int peer_id = std::stoi(recv_msgs[0].to_string());
        auto metadata = decerealize<Metadata>((char*) recv_msgs[1].data(), recv_msgs[1].size());
        torch::Tensor tensor = torch::empty(
            {metadata->num_tokens(), metadata->token_hidden_dim()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU)
        );
        // auto tensor_buf = (uintptr_t) std::malloc(metadata->num_element() * metadata->get_datatype_size());

        this->peer_channels[peer_id]->recv((uintptr_t)tensor.data_ptr(), *metadata);
        
        int tokens = this->process_batch(tensor, metadata);
        token_processed += tokens;
        iter ++;
        if (iter >= 500) {
            long long cur_time_ms = t_now_high();
            int elapsed_time_ms = static_cast<int>(cur_time_ms - start_timestamp_ms) ;
            int token_throughput = token_processed * 1000 / elapsed_time_ms;

            DMOE_LOG(INFO) << "token throughput: " << token_throughput << " tokens/s" << LEND;
            iter = 0;
            token_processed = 0;
            start_timestamp_ms = cur_time_ms;
        }
    }
}

int Sampler::process_batch(torch::Tensor tensor, metadata_t meta) {
    std::lock_guard<std::mutex> _(this->result_lock);
    // DMOE_LOG(WARNING) << "processing batch:" << *meta << ", with shape: " << tensor.sizes()[0] << ", " << tensor.sizes()[1] << LEND;
    int num_tokens = meta->req_ids.size();
    auto cur_time_ms = t_now_high();
    for (int i = 0; i < num_tokens; i ++) {
        int rid = meta->req_ids[i];
        this->_active_token_count ++;
        if (slo_stats.find(rid) != slo_stats.end()) {
            slo_stats[rid].t_tokens.push_back(cur_time_ms);
        } else {
            slo_stats[rid] = SloStat {rid, cur_time_ms, cur_time_ms, 0, {}};
        }
    }
    this->step_infos.emplace_back(num_tokens, cur_time_ms);
    // DMOE_LOG(INFO) << "sampler processed tokens " << num_tokens << ", at timestamp " << cur_time_ms << LEND;
    // DMOE_LOG(INFO) << "sampler processed tokens " << this->_active_token_count << LEND;
    auto finish_indices = meta->get_finished_indices();
    for (auto &idx: finish_indices) {
        int rid = meta->req_ids[idx];
        // DMOE_LOG(INFO) << "finished req " << rid << LEND;
        this->_active_token_count --;
        ASSERT (slo_stats.find(rid) != slo_stats.end());
        slo_stats[rid].t_decode = cur_time_ms;
        finished_seqs.insert(rid);
    }
    return num_tokens;
}

std::vector<SloStat> Sampler::fetch_finished_slo_stats() {
    std::lock_guard<std::mutex> _(this->result_lock);
    std::vector<SloStat> res {};
    for (auto &x: finished_seqs) {
        res.emplace_back(slo_stats[x]);
        slo_stats.erase(x);
    }
    finished_seqs.clear();
    return res;
}

std::vector<SamplerStepInfo> Sampler::fetch_step_infos() {
    std::lock_guard<std::mutex> _(this->result_lock);
    return std::move(step_infos);
}

void Sampler::reset() {
    std::lock_guard<std::mutex> _(this->result_lock);
    if (step_infos.size() > 0) {
        step_infos.clear();
    }
    if (slo_stats.size() > 0) {
        slo_stats.clear();
    }
    if (finished_seqs.size() > 0) {
        finished_seqs.clear();
    }
    this->_active_token_count = 0;
}

std::map<int, SloStat> Sampler::wait_slo_stats(int n_request) {
    std::lock_guard<std::mutex> _(this->result_lock);
    if (finished_seqs.size() < n_request)
        return {};
    finished_seqs.clear();
    return slo_stats;
}

void Sampler::start() {
    MuExpertDispatcher::start();
}

Tokenizer::Tokenizer(int device_id, 
              ParallelConfig cfg,
              std::vector<Channel_t> channels, 
              std::vector<ChannelInfo> out_channel_infos):
    MuExpertDispatcher({}, device_id, ParallelConfig(1, 1, cfg.dp, 1, {}), channels, out_channel_infos) {
}

void Tokenizer::put_request(int req_id, int init_prefill_len, torch::Tensor tensor, int dp_rank) {
    // TODO(hogura|20241007): set the first attn
    ASSERT (tensor.dim() == 2);
    std::vector<size_t> shape{tensor.size(0), tensor.size(1)};
    auto meta_t = std::make_shared<Metadata>(Metadata {
        shape, 
        "bf16", 
        /*layer_id=*/ 0, 
        /*req_id=*/ {req_id},
        /*exp_ids=*/ {-1},
        /*attn_ids=*/ {dp_rank},
        /*init_prefill_lens=*/ {init_prefill_len},
    });
    this->put(TensorBatch {tensor.clone().detach(), meta_t}, 0);

}

void Tokenizer::start() {
    MuExpertDispatcher::start();
}