#include "embedding.h"
#include "constants.h"
#include "logging.h"
#include "utils.hpp"

#include "zmq.hpp"
#include "zmq_addon.hpp"

#include <thread>

Sampler::Sampler(int device_id, 
                std::vector<Channel_t> in_channels, 
                std::vector<Channel_t> out_channels,
                std::vector<ChannelInfo> out_channel_infos):
    MuExpertDispatcher(
        /*layer_ids=*/ {0}, 
        device_id, 
        ParallelConfig(1, 1, 1),
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
    this->recv_mq.bind(get_zmq_addr(device_id));
    for (int i = 0; i < this->channels.size(); i ++)
        this->peer_mq[i].connect(get_zmq_addr(this->channels[i]->get_peer_id()));

    while (!this->end_flag) {
        DMOE_LOG(DEBUG) << "sampler receiving msg ..." << LEND;
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_result_t result =
            zmq::recv_multipart(this->recv_mq, std::back_inserter(recv_msgs));
        
        DMOE_LOG(DEBUG) << "sampler got msg !!!" << LEND;
        ASSERT(*result == 2);
        int peer_id = std::stoi(recv_msgs[0].to_string());
        auto metadata = decerealize<Metadata>((char*) recv_msgs[1].data(), recv_msgs[1].size());
        torch::Tensor tensor = torch::empty(
            {metadata->num_element()}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU)
        );
        // auto tensor_buf = (uintptr_t) std::malloc(metadata->num_element() * metadata->get_datatype_size());

        this->peer_channels[peer_id]->recv((uintptr_t)tensor.data_ptr(), *metadata);
        
        this->process_batch(tensor, metadata);
    }
}

int Sampler::_get_attn_channel(int req_id, int layer_id) {
    // TODO(hogura|20241007): no attn replicas yet
    return 0;
}

void Sampler::process_batch(torch::Tensor tensor, metadata_t meta) {
    std::lock_guard<std::mutex> _(this->result_lock);
    DMOE_LOG(DEBUG) << "processing batch:" << *meta << LEND;

    // Step 1. select finished & unfinished batches
    std::vector<int> continue_ids;

    for (int i = 0; i < meta->req_ids.size(); i ++) {
        int rid = meta->req_ids[i];
        output_lens[rid] ++;

        if (meta->prefill_poss[i] == -1) {
            // at decode phase
            if (eos_seqs.find(rid) == eos_seqs.end()) {
                // Not marked as finished, can continue.
                continue_ids.push_back(i);
                slo_stats[rid].t_tokens.push_back(clock());
            }
            else {
                // Finished, end.
                finished_seqs.insert(rid);
                eos_seqs.erase(rid);
                DMOE_LOG(INFO) << "Request " << rid << " ended, generated " 
                          << output_lens[rid] << " tokens." << LEND;
            }
        } else {
            // at prefill phase
            ASSERT (slo_stats.find(rid) == slo_stats.end());
            continue_ids.push_back(i);
            slo_stats[rid] = SloStat {rid, clock(), 0, {}};
        }
    }

    // Step 2. update metadata
    meta->layer_id = 0;
    Metadata new_meta = meta->at(continue_ids);
    for (int i = 0; i < new_meta.req_ids.size(); i ++) {
        // !NOTE(hogura|20241007): 
        // 1. no chunked prefill, directly prefill -> decode;
        // 2. no attn replica, first_attn_id = 0
        if (new_meta.prefill_poss[i] != -1) {
            new_meta.prefill_poss[i] = -1;
        }
    }

    // Step 3. send batches
    // TODO(hogura|20241007): attention id control
    DMOE_LOG(DEBUG) << "sampler send once with new meta: " << new_meta << LEND;

    _send_once(TensorBatch{
        torch_tensor_slice(tensor, continue_ids),
        std::make_shared<Metadata>(new_meta)
    });

    // Step 4. sample tokens & marked finished
    for (int i: continue_ids) {
        int req_id = meta->req_ids[i];
        int token = sample(tensor_at((uint64_t)tensor.data_ptr(), meta, i), meta);
        if (check_finished(token, req_id)) {
            eos_seqs.insert(req_id);
            slo_stats[req_id].t_decode = clock();
        }
        // TODO(hogura|20241007): send the generated tokens back to master node
    }

    DMOE_LOG(INFO) << "sampler processed one batch" << LEND;
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

std::map<int, SloStat> Sampler::wait_slo_stats(int n_request) {
    std::lock_guard<std::mutex> _(this->result_lock);
    if (finished_seqs.size() < n_request)
        return {};
    finished_seqs.clear();
    return slo_stats;
}

int Sampler::sample(uintptr_t buf, metadata_t meta) {
    // TODO(hogura|20241007): implement a real sampling function
    return 233;
}

bool Sampler::check_finished(int token, int req_id) {
    return token == EOS_TOKEN_ID || this->output_lens[req_id] >= MAX_OUTPUT_LEN;
}

void Sampler::start() {
    MuExpertDispatcher::start();
}

Tokenizer::Tokenizer(int device_id, 
              std::vector<Channel_t> channels, 
              std::vector<ChannelInfo> out_channel_infos):
    MuExpertDispatcher({}, device_id, ParallelConfig(1, 1, 1), channels, out_channel_infos) {
}

void Tokenizer::put_request(int req_id, torch::Tensor tensor) {
    // TODO(hogura|20241007): set the first attn
    ASSERT (tensor.dim() == 2);
    std::vector<size_t> shape{tensor.size(0), tensor.size(1)};
    auto meta_t = std::make_shared<Metadata>(Metadata {
        shape, 
        "fp16", 
        /*layer_id=*/ 0, 
        /*req_id=*/ {req_id},
        /*exp_ids=*/ {-1},
        /*prefill_pos=*/ {0},
        /*prompt_lens=*/ {}
    });
    this->put(TensorBatch {tensor.clone().detach(), meta_t}, 0);
}

void Tokenizer::start() {
    MuExpertDispatcher::start();
}