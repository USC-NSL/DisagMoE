#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <condition_variable>

#include "comm.h"
#include "zmq.hpp"

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

    void init_cuda_device();

    void terminate();

    int get_device_id();

};


class MuDispatcher: public MuHelper {
    
protected:
    char device_id_str[3];

    std::queue<TensorBatch> send_queue;
    std::mutex mtx;
    std::condition_variable cv;

    // ctx must be ahead of mq
    std::vector<zmq::context_t> peer_ctx;
    std::vector<zmq::socket_t> peer_mq;

    virtual void _send_once(TensorBatch batch) = 0;

    void _send_batch(int cid, uintptr_t buf, const Metadata& meta);

    void run() override;

public:
    MuDispatcher(std::vector<int> layer_ids, int device_id, std::vector<Channel_t> channels);

    void put(const TensorBatch &batch);
};


class MuAttnDispatcher: public MuDispatcher {

protected:
    void _send_once(TensorBatch batch) override;

public:
    MuAttnDispatcher(std::vector<int> layer_ids, 
                     int device_id, 
                     std::vector<Channel_t> channels={});
};

class MuExpertDispatcher: public MuDispatcher {
protected:
    std::vector<ChannelInfo> channel_infos;
    std::vector<int> attn_channel;

    void _send_once(TensorBatch batch) override;
    int _get_attn_channel(int req_id, int layer_id);

public:
    MuExpertDispatcher(std::vector<int> layer_ids, 
                       int device_id, 
                       std::vector<Channel_t> channels={},
                       std::vector<ChannelInfo> channel_infos={});
};

class MuPool: public MuHelper {
protected:
    bool is_attn;
    std::vector<Channel_t> peer_channels;

    // ctx must be ahead of mq
    zmq::context_t ctx;
    zmq::socket_t mq;

    std::vector<std::vector<TensorBatch>> data_queue;
    std::vector<int> inner_layer_id;
    std::vector<std::mutex> layer_mutex;

    std::mutex request_mutex;
    std::condition_variable request_cv;
    int cur_request_count;

public:
    MuPool(std::vector<int> layer_ids,
           int device_id,
           std::vector<Channel_t> channels,
           bool is_attn = false);

    void run() override;

    void wait_for_new_requests();

    std::vector<TensorBatch> fetch_largest_batch();
};

typedef std::shared_ptr<MuPool> MuPool_t;
typedef std::shared_ptr<MuDispatcher> MuDispatcher_t;