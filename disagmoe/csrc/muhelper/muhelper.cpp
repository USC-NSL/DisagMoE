#include "datatypes.h"
#include "muhelper.h"
#include "comm.h"

#include <condition_variable>
#include <mutex>
#include <queue>

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
    puts("sent a batch.");
    // TODO(hogura|20240926)
}