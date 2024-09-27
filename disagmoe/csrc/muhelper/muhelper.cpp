#include <condition_variable>
#include <mutex>
#include <queue>

#include "datatypes.h"
#include "muhelper.h"
#include "comm.h"
#include "utils.hpp"

// MuHelper

MuHelper::MuHelper(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels): 
    layer_ids(layer_ids), device_id(device_id), channels(channels), end_flag(false) {}

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
    MuHelper(layer_ids, device_id, channels) 
    {}

void MuDispatcher::run() {
    while (!this->end_flag) {
        std::unique_lock<std::mutex> lock(this->mtx);
        this->cv.wait(lock, [&] { return !this->send_queue.empty(); });
        auto batch = this->send_queue.front();
        this->send_queue.pop();
        this->_send_once(batch);
    }
}

void MuDispatcher::put(const TensorBatch &batch) {
    puts("MuDispatcher::put");
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
    for (int i = 0; i < batch.metadata->shape[0];) {
        int j = i + 1;
        int eid = batch.metadata->infos[i].exp_id;
        while (j < n && batch.metadata->infos[j].exp_id == eid)
            j ++;
        auto buf = tensor_at(batch.data, *batch.metadata, i);
        this->channels[eid]->send(buf, batch.metadata->slice(i, j));
    }

    puts("sent a batch.");
}


MuPool::MuPool(int device_id,
    std::vector<Channel_t> channels,
    bool is_attn): MuHelper({}, device_id, channels), is_attn(is_attn) {
    
}

void MuPool::run() {
    while (this->end_flag) {
        puts("fetching metadata ...");

    }
}

void MuPool::wait_for_new_requests() {
    
}

std::vector<TensorBatch> MuPool::fetch_largest_batch() {

}