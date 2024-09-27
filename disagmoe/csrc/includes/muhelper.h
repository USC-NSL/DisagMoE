#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <condition_variable>

#include "comm.h"

class MuHelper {

protected:

    std::vector<int> layer_ids;

    int device_id;
    bool end_flag;

    std::thread thread;
    std::vector<Channel_t> channels;

    virtual void run() = 0;

public:
    MuHelper(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels);

    void start();

    void terminate();

};


class MuDispatcher: public MuHelper {
    
protected:
    std::queue<TensorBatch> send_queue;
    std::mutex mtx;
    std::condition_variable cv;

    virtual void _send_once(TensorBatch batch) = 0;

    void run() override;

public:
    MuDispatcher(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels);

    void put(const TensorBatch &batch);
};


class MuAttnDispatcher: public MuDispatcher {

protected:
    void _send_once(TensorBatch batch) override;

public:
    MuAttnDispatcher(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels={});
};