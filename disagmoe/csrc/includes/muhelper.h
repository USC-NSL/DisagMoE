#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <condition_variable>

#include "datatypes.hpp"
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
    virtual int _get_attn_channel(int req_id, int layer_id);

public:
    MuExpertDispatcher(std::vector<int> layer_ids, 
                       int device_id, 
                       std::vector<Channel_t> channels={},
                       std::vector<ChannelInfo> channel_infos={});
    
    void debug_put(TensorBatch batch);
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

    std::mutex request_mutex;
    std::mutex batch_mutex;

    std::condition_variable request_cv;
    int cur_request_count;

    int largest_batch_size_{0};
    int largest_batch_layer_id_{-1};
    std::vector<int> tokens_per_layer_;

    virtual int tokens_in_layer(int lid);

public:
    MuPool(std::vector<int> layer_ids,
           int device_id,
           std::vector<Channel_t> channels,
           bool is_attn = false);

    void run() override;

    void wait_for_new_requests();

/* 

for attention, consider waiting sequences,

1.first layer consider add waiting seqs, count(can_alloc())

2. later layers pick largest running batch, use token number

*/
    std::vector<TensorBatch> fetch_largest_batch();
};

typedef std::shared_ptr<MuPool> MuPool_t;
typedef std::shared_ptr<MuDispatcher> MuDispatcher_t;

class MuAttentionPool: public MuPool {

private:

    std::vector<std::vector<AttentionBatch>> attn_data_queue;

    AttentionBatch pack_attn_batch(uintptr_t, metadata_t);

    int tokens_in_layer(int lid) override;

public:

    MuAttentionPool(std::vector<int> layer_ids,
           int device_id,
           std::vector<Channel_t> channels);

    std::vector<AttentionBatch> fetch_largest_batch();

    void run() override;

};