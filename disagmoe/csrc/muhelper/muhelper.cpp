#include <condition_variable>
#include <sstream>
#include <cstdlib>
#include <string>
#include <mutex>
#include <queue>
#include <ctime>

#include "datatypes.hpp"
#include "muhelper.h"
#include "comm.h"
#include "utils.hpp"

#include "zmq.hpp"
#include "zmq_addon.hpp"

#include <cereal/archives/binary.hpp>

// MuHelper

MuHelper::MuHelper(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    layer_ids(layer_ids), device_id(device_id), channels(channels), end_flag(false) {
        puts("init muhelper.");
    }

void MuHelper::start() {
    puts("start");
    this->thread = std::thread(
        [&](MuHelper* helper) { helper->run(); }, 
        this
    );
}

void MuHelper::terminate() {
    this->end_flag = true;
    this->thread.join();
}

// MuDispatcher

MuDispatcher::MuDispatcher(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    MuHelper(layer_ids, device_id, channels), 
    ctx(channels.size()), 
    mq(this->ctx, zmq::socket_type::push) {
    
}

std::string _serialize(metadata_t metadata) {
    // use cereal to serialize metadata
    std::stringstream ss;
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(*metadata);
    return ss.str();
}

metadata_t _deserialize(char* buf, size_t n) {
    std::string buffer(buf, n);
    std::istringstream ss(buffer);
    cereal::BinaryInputArchive iarchive(ss);
    Metadata result;
    iarchive(result);
    puts("end of cereal");
    return std::make_shared<Metadata>(result);
}

void MuDispatcher::run() {
    this->mq.connect("tcp://127.0.0.1:24927");
    int i = 0;
    char device_id[2] = "";
    sprintf(device_id, "%d", this->device_id, device_id);
    while (!this->end_flag && i < 1024) {
        std::unique_lock<std::mutex> lock(this->mtx);
        this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
        auto batch = this->send_queue.front();
        this->send_queue.pop();
        
        auto data = _serialize(batch.metadata);
        this->mq.send(zmq::str_buffer(device_id), zmq::send_flags::sndmore);
        printf("sending data with %u bytes\n", data.size());
        this->mq.send(zmq::buffer(data.c_str(), data.size()));
        this->_send_once(batch);
        i += 1;
    }
}

void MuDispatcher::put(const TensorBatch &batch) {
    // puts("MuDispatcher::put");
    std::lock_guard<std::mutex> lock(this->mtx);
    this->send_queue.push(batch);
    this->cv.notify_one();
}

// MuAttnDispatcher

MuAttnDispatcher::MuAttnDispatcher(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    MuDispatcher(layer_ids, device_id, channels) 
    {}

void MuAttnDispatcher::_send_once(TensorBatch batch) {
    puts("sending a batch.");

    int n = batch.metadata->shape[0];
    for (int i = 0; i < n;) {
        int j = i + 1;
        int eid = batch.metadata->infos[i].exp_id;
        while (j < n && batch.metadata->infos[j].exp_id == eid)
            j ++;
        auto buf = tensor_at(batch.data, *batch.metadata, i);
        this->channels[eid]->send(buf, batch.metadata->slice(i, j));
        i = j;
    }

    puts("sent a batch.");
}


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

    // for (auto c: channels) {
    //     char peer_id[2];
    //     sprintf(peer_id, "%d", c->get_peer_id());
    //     this->mq.set(zmq::sockopt::subscribe, peer_id);
    // }

    this->layer_mutex = std::vector<std::mutex>(this->data_queue.size());

    this->cur_request_count = 0;
}

void MuPool::run() {
    this->mq.bind("tcp://127.0.0.1:24927");

    auto last = clock();
    auto start = last;

    while (!this->end_flag) {
        puts("fetching metadata ...");
        
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_result_t result =
            zmq::recv_multipart(this->mq, std::back_inserter(recv_msgs));
            
        puts("get a msg");
        assert(*result == 2);
        printf("get data with size: %u\n", recv_msgs[1].size());

        int peer_id = std::stoi(recv_msgs[0].to_string());
        auto metadata = _deserialize((char*) recv_msgs[1].data(), recv_msgs[1].size());
        auto tensor_buf = alloc_cuda_tensor(metadata->num_element(), this->device_id);
        this->channels[peer_id]->recv(tensor_buf, *metadata);

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

        puts("!!!!! unlocked request_mutex");

        // i += 1;
        // printf("info len: %d\n", metadata.infos.size());
        // float single_step = 1.0 * (clock() - last) / CLOCKS_PER_SEC;
        // float total_step =  1.0 * (clock() - start) / CLOCKS_PER_SEC;
        // printf("i %d, elapsed %f, tp %f\n", i, single_step, total_step, i / total_step);
        // last = clock();
    }
}

void MuPool::wait_for_new_requests() {
    puts("MuPool waiting for new requests");
    std::unique_lock<std::mutex> lock(this->request_mutex);
    if (this->cur_request_count > 0) {
        lock.unlock();
        return;
    }
    this->request_cv.wait(lock, [&] { return this->cur_request_count > 0; });
    lock.unlock();
    puts("MuPool got new requests.");
}

std::vector<TensorBatch> MuPool::fetch_largest_batch() {
    // TODO(hogura|20240930): only considering decode first
    puts("fetching largest batch");
    size_t max_batch_size = 0;
    int id = -1;
    for (size_t i = 0; i < this->data_queue.size(); i ++) {
        printf("checking data_queue %u\n", i);
        std::lock_guard<std::mutex> lock(this->layer_mutex[i]);
        if (max_batch_size < this->data_queue[i].size()) {
            max_batch_size = this->data_queue[i].size();
            id = i;
        }
        printf("checked data_queue %u\n", i);
    }
    printf("==> fecthed largest batch @ %d\n", id);
    if (id == -1) {
        puts("no batch available");
        return {};
    }

    printf("now fetching ...\n");
    std::vector<TensorBatch> result;
    {
        std::lock_guard<std::mutex> lock(this->layer_mutex[id]);

        result = this->data_queue[id];
        this->data_queue[id].clear();
    }

    printf("update request count ...\n");
    {
        std::lock_guard<std::mutex> lock(this->request_mutex);
        this->cur_request_count -= result.size();
        assert(this->cur_request_count >= 0);
    }

    printf("!!! fecthed largest batch @ %d\n", id);
    return result;
}