#include <condition_variable>
#include <cstdlib>
#include <string>
#include <mutex>
#include <queue>
#include <ctime>

#include "datatypes.hpp"
#include "muhelper.h"
#include "comm.h"
#include "utils.hpp"
#include "logging.h"
#include "constants.h"

#include "zmq.hpp"
#include "zmq_addon.hpp"

#include <cereal/archives/binary.hpp>

// MuHelper

MuHelper::MuHelper(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    layer_ids(layer_ids), device_id(device_id), channels(channels), end_flag(false) {
        LOG(INFO) << "init muhelper@" << device_id << LEND;
    }

void MuHelper::start() {
    LOG(INFO) << "muhelper@" << device_id << " start" << LEND;
    this->thread = std::thread(
        [&](MuHelper* helper) {
            helper->init_cuda_device();
            helper->run(); 
        }, 
        this
    );
}

int MuHelper::get_device_id() {
    return device_id;
}

void MuHelper::terminate() {
    this->end_flag = true;
    this->thread.join();
}

void MuHelper::init_cuda_device() {
    #ifndef D_ENABLE_RAY
    CUDACHECK(cudaSetDevice(this->device_id));
    #endif
}

// MuDispatcher

MuDispatcher::MuDispatcher(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    MuHelper(layer_ids, device_id, channels), 
    peer_ctx(channels.size()),
    peer_mq(channels.size()) {
    sprintf(this->device_id_str, "%d", this->device_id);
    for (int i = 0; i < channels.size(); i ++) {
        peer_ctx[i] = zmq::context_t(1);
        peer_mq[i] = zmq::socket_t(peer_ctx[i], zmq::socket_type::push);
    }
}

void MuDispatcher::_send_batch(int cid, uintptr_t buf, const Metadata& meta) {
    auto data = cerealize(std::make_shared<Metadata>(meta));
    this->peer_mq[cid].send(zmq::str_buffer(this->device_id_str), zmq::send_flags::sndmore);
    this->peer_mq[cid].send(zmq::buffer(data.c_str(), data.size()));
    this->channels[cid]->send(buf, meta);
}

void MuDispatcher::run() {
    for (int i = 0; i < this->channels.size(); i ++)
        this->peer_mq[i].connect(get_zmq_addr(this->channels[i]->get_peer_id()));

    while (!this->end_flag) {
        std::unique_lock<std::mutex> lock(this->mtx);
        this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
        auto batch = this->send_queue.front();
        this->send_queue.pop();
        this->_send_once(batch);
    }
}

void MuDispatcher::put(const TensorBatch &batch) {
    std::lock_guard<std::mutex> lock(this->mtx);
    this->send_queue.push(batch);
    this->cv.notify_one();
}

/*
    MuAttnDispatcher
*/

MuAttnDispatcher::MuAttnDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    std::vector<Channel_t> channels): 
        MuDispatcher(layer_ids, device_id, channels) {
    
}

void MuAttnDispatcher::_send_once(TensorBatch batch) {
    LOG(DEBUG) << "sending a batch." << LEND;

    int n = batch.metadata->shape[0];
    for (int i = 0; i < n;) {
        int j = i + 1;
        int eid = batch.metadata->infos[i].exp_id;
        while (j < n && batch.metadata->infos[j].exp_id == eid)
            j ++;
        auto buf = tensor_at(batch.data, *batch.metadata, i);
        this->_send_batch(
            eid,
            buf,
            batch.metadata->slice(i, j)
        );
        i = j;
    }

    LOG(DEBUG) << "sent a batch." << LEND;
}

/*
    MuExpertDispatcher
*/

MuExpertDispatcher::MuExpertDispatcher(
    std::vector<int> layer_ids, 
    int device_id, 
    std::vector<Channel_t> channels,
    std::vector<ChannelInfo> channel_infos): 
        MuDispatcher(layer_ids, device_id, channels),
        channel_infos(channel_infos) {
    int max_layer = 0;
    for (auto i: layer_ids)
        max_layer = std::max(i, max_layer);
    this->attn_channel = std::vector<int>(max_layer + 1, -1);

    if (channels.size() == channel_infos.size()) {
        for (size_t i = 0; i < channels.size(); i ++) {
            // TODO(hogura|20240930): currently, only support #attn_replica=1
            assert(channel_infos[i].attn_layer_ids.size() <= 1);
            this->attn_channel[
                channel_infos[i].attn_layer_ids[0]
            ] = i;
        }
    }
}

int MuExpertDispatcher::_get_attn_channel(int req_id, int layer_id) {
    assert(this->attn_channel[layer_id] != -1);
    return this->attn_channel[layer_id];
}

void MuExpertDispatcher::debug_put(TensorBatch batch) {
    _send_once(batch);
}

void MuExpertDispatcher::_send_once(TensorBatch batch) {
    LOG(DEBUG) << "expert sending a batch" << LEND;
    auto meta = batch.metadata;
    auto layer_id = meta->layer_id;
    std::vector<int> chans;
    for (auto &info: meta->infos) {
        chans.push_back(_get_attn_channel(info.req_id, meta->layer_id));
    }

    auto batches = group_by<int, std::less<int>>(batch.data, *meta, chans);
    LOG(DEBUG) << "grouped channels" << LEND;

    for (auto &[channel, sub_batch]: batches) {
        this->_send_batch(
            channel,
            sub_batch.data,
            *sub_batch.metadata
        );
    }
    LOG(DEBUG) << "expert sent a batch" << LEND;
}

/*
    MuPool
*/

MuPool::MuPool(
    std::vector<int> layer_ids, 
    int device_id,
    std::vector<Channel_t> channels,
    bool is_attn): 
    MuHelper(layer_ids, device_id, channels), 
    is_attn(is_attn),
    ctx(channels.size()),
    mq(ctx, zmq::socket_type::pull) {

    int max_layer_id = 0;
    for (auto id: layer_ids)
        max_layer_id = std::max(max_layer_id, id);
    this->inner_layer_id = std::vector<int>(max_layer_id + 1);
    this->data_queue = std::vector<std::vector<TensorBatch>>(layer_ids.size());
    for (size_t i = 0; i < layer_ids.size(); i ++)
        this->inner_layer_id[layer_ids[i]] = i;

    this->layer_mutex = std::vector<std::mutex>(this->data_queue.size());

    this->cur_request_count = 0;

    int max_peer_id = 0;
    for (auto c: channels)
        max_peer_id = std::max(max_peer_id, c->get_peer_id());
    this->peer_channels = std::vector<Channel_t>(max_peer_id + 1);
    for (size_t i = 0; i < channels.size(); i ++) {
        int id = channels[i]->get_peer_id();
        assert(this->peer_channels[id].get() == nullptr);
        this->peer_channels[ channels[i]->get_peer_id() ] = channels[i];
    }
}

void MuPool::run() {
    if (this->channels.empty()) {
        LOG(WARNING) << this->device_id << " has no channels, exit MuPool." << LEND;
        return;
    }
    this->mq.bind(get_zmq_addr(this->device_id));

    auto last = clock();
    auto start = last;

    while (!this->end_flag) {
        LOG(DEBUG) << "fetching a msg ..." << LEND;
        
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_result_t result =
            zmq::recv_multipart(this->mq, std::back_inserter(recv_msgs));
            
        LOG(DEBUG) << "got a msg!" << LEND;
        assert(*result == 2);

        int peer_id = std::stoi(recv_msgs[0].to_string());
        auto metadata = decerealize((char*) recv_msgs[1].data(), recv_msgs[1].size());
        auto tensor_buf = alloc_cuda_tensor(metadata->num_element(), this->device_id);
        LOG(DEBUG) << "calling NCCL recv from " << peer_id << " metadata= " << *metadata << LEND;
        this->peer_channels[peer_id]->recv(tensor_buf, *metadata);

        int lid = this->inner_layer_id[metadata->layer_id];

        {
            std::lock_guard<std::mutex> lock(this->layer_mutex[lid]);
            this->data_queue[lid].push_back((TensorBatch) {
                tensor_buf,
                metadata
            });
        }

        {
            std::lock_guard<std::mutex> lock(this->request_mutex);
            // TODO(hogura|20240930): should modify this `1` with `metadata.num_tokens`
            this->cur_request_count += 1;
            this->request_cv.notify_all();
        }
    }
}

void MuPool::wait_for_new_requests() {
    LOG(INFO) << "MuPool waiting for new requests" << LEND;
    std::unique_lock<std::mutex> lock(this->request_mutex);
    if (this->cur_request_count > 0) {
        lock.unlock();
        return;
    }
    this->request_cv.wait(lock, [&] { return this->cur_request_count > 0; });
    lock.unlock();
    LOG(INFO) << "MuPool got new requests." << LEND;
}

std::vector<TensorBatch> MuPool::fetch_largest_batch() {
    // TODO(hogura|20240930): only considering decode first
    LOG(INFO) << "fetching largest batch" << LEND;
    size_t max_batch_size = 0;
    int id = -1;
    for (size_t i = 0; i < this->data_queue.size(); i ++) {
        std::lock_guard<std::mutex> lock(this->layer_mutex[i]);
        if (max_batch_size < this->data_queue[i].size()) {
            max_batch_size = this->data_queue[i].size();
            id = i;
        }
    }
    
    if (id == -1) {
        LOG(INFO) << "No available batch" << LEND;
        return {};
    }
    LOG(INFO) << "Fetched " << id << " layer" << LEND;

    std::vector<TensorBatch> result;
    {
        std::lock_guard<std::mutex> lock(this->layer_mutex[id]);

        result = this->data_queue[id];
        this->data_queue[id].clear();
    }

    {
        std::lock_guard<std::mutex> lock(this->request_mutex);
        this->cur_request_count -= result.size();
        assert(this->cur_request_count >= 0);
    }

    return result;
}