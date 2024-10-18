#include "embedding.h"
#include "constants.h"
#include "logging.h"
#include "utils.hpp"

#include "zmq.hpp"
#include "zmq_addon.hpp"

Sampler::Sampler(int device_id, 
                std::vector<Channel_t> in_channels, 
                std::vector<Channel_t> out_channels,
                std::vector<ChannelInfo> out_channel_infos):
    MuExpertDispatcher(
        /*layer_ids=*/ {0}, 
        device_id, 
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
        LOG(INFO) << "inited sampler" << LEND;
    }

void Sampler::run() {
    this->recv_mq.bind(get_zmq_addr(device_id));
    for (int i = 0; i < this->channels.size(); i ++)
        this->peer_mq[i].connect(get_zmq_addr(this->channels[i]->get_peer_id()));

    while (!this->end_flag) {
        LOG(DEBUG) << "sampler receiving msg ..." << LEND;
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_result_t result =
            zmq::recv_multipart(this->recv_mq, std::back_inserter(recv_msgs));
        
        LOG(DEBUG) << "sampler got msg !!!" << LEND;
        ASSERT(*result == 2);
        int peer_id = std::stoi(recv_msgs[0].to_string());
        auto metadata = decerealize((char*) recv_msgs[1].data(), recv_msgs[1].size());
        auto tensor_buf = (uintptr_t) std::malloc(metadata->num_element() * metadata->get_datatype_size());

        this->peer_channels[peer_id]->recv(tensor_buf, *metadata);
        
        this->process_batch(tensor_buf, metadata);
    }
}

int Sampler::_get_attn_channel(int req_id, int layer_id) {
    // TODO(hogura|20241007): no attn replicas yet
    return 0;
}

void Sampler::process_batch(uintptr_t buf, metadata_t meta) {
    LOG(DEBUG) << "processing batch:" << *meta << LEND;

    // Step 1. select finished & unfinished batches
    std::vector<int> continue_ids;

    for (int i = 0; i < meta->infos.size(); i ++) {
        auto &info = meta->infos[i];
        int rid = info.req_id;
        output_lens[rid] ++;

        if (info.prefill_pos == -1) {
            // at decode phase
            if (finished_seqs.find(rid) == finished_seqs.end()) {
                // Not marked as finished, can continue.
                continue_ids.push_back(i);
            }
            else {
                // Finished, end.
                LOG(INFO) << "Request " << rid << " ended, generated " 
                          << output_lens[rid] << " tokens." << LEND;
            }
        } else {
            // at prefill phase
            continue_ids.push_back(i);
        }
    }

    // Step 2. update metadata
    meta->layer_id = 0;
    Metadata new_meta = meta->at(continue_ids);
    for (auto &info: new_meta.infos) {
        // !NOTE(hogura|20241007): 
        // 1. no chunked prefill, directly prefill -> decode;
        // 2. no attn replica, first_attn_id = 0
        if (info.prefill_pos != -1) {
            info.prefill_pos = -1;
            info.first_attn_id = 0;
        }
    }

    // Step 3. send batches
    // TODO(hogura|20241007): attention id control
    LOG(DEBUG) << "sampler send once with new meta: " << new_meta << LEND;
    _send_once(TensorBatch{
        tensor_slice(buf, meta, continue_ids, /*on_gpu=*/ false),
        std::make_shared<Metadata>(new_meta)
    });

    // Step 4. sample tokens & marked finished
    for (int i: continue_ids) {
        auto &info = meta->infos[i];
        int token = sample(tensor_at(buf, meta, i), meta);
        if (check_finished(token, info.req_id)) {
            finished_seqs.insert(info.req_id);
        }
        // TODO(hogura|20241007): send the generated tokens back to master node
    }

    LOG(INFO) << "sampler processed one batch" << LEND;
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
    MuExpertDispatcher({}, device_id, channels, out_channel_infos) {
    req_count = 0;
}

void Tokenizer::put_request(uintptr_t buf, std::vector<size_t> shape) {
    req_count ++;
    // TODO(hogura|20241007): set the first attn
    auto meta_t = std::make_shared<Metadata>(Metadata {
        shape, 
        "fp16", 
        /*layer_id=*/ 0, 
        { (TokenMetadata) {
            req_count, 
            /*exp_id=*/ -1, 
            /*first_attn_id=*/ 0, 
            /*prefill_pos=*/ 0} },
        /*prompt_lens=*/ {}
    });
    this->put(TensorBatch {buf, meta_t});
}

void Tokenizer::start() {
    MuExpertDispatcher::start();
}