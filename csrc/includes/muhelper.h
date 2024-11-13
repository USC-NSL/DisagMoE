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

    std::queue<std::pair<TensorBatch, int>> send_queue;
    std::mutex mtx;
    std::condition_variable cv;

    // ctx must be ahead of mq
    std::vector<zmq::context_t> peer_ctx;
    std::vector<zmq::socket_t> peer_mq;

    ParallelConfig cfg;

    virtual void _send_once(TensorBatch batch) = 0;

    void _send_batch(int cid, uintptr_t buf, const Metadata& meta);

    void run() override;

public:
    MuDispatcher(std::vector<int> layer_ids, 
                 int device_id, 
                 ParallelConfig cfg, 
                 std::vector<Channel_t> channels);

    void put(const TensorBatch &batch, int rank = 0);
};


class MuAttnDispatcher: public MuDispatcher {

protected:
    std::vector<int> exp_channels;
    int max_exp_id;

    void _send_once(TensorBatch batch) override;

    int _encode(int exp_layer_id, int exp_id) const;

    int _get_rank(int exp_id) const;

public:
    MuAttnDispatcher(std::vector<int> layer_ids, 
                     int device_id, 
                     ParallelConfig cfg,
                     std::vector<Channel_t> channels={},
                     const std::vector<ChannelInfo> &out_channel_infos={});
};

class MuExpertDispatcher: public MuDispatcher {
protected:
    std::vector<ChannelInfo> channel_infos;
    std::vector<int> attn_channel;
    int sampler_channel_id;

    void _send_once(TensorBatch batch) override;
    virtual int _get_attn_channel(int req_id, int layer_id);

public:
    MuExpertDispatcher(std::vector<int> layer_ids, 
                       int device_id, 
                       ParallelConfig cfg,
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
    int cur_request_count{0};

    int largest_batch_size_{0};
    int largest_batch_layer_id_{-1};
    std::vector<int> tokens_per_layer_;

    virtual int tokens_in_layer(int lid);

    void recv_metadata(int &peer_id, metadata_t &meta);
    void recv_tensor(int peer_id, uintptr_t tensor_buf, metadata_t &meta);
    virtual void process_batch(torch::Tensor tensor, metadata_t &meta);

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

    void maintain_largest_batch();
};

typedef std::shared_ptr<MuPool> mu_pool_t;
typedef std::shared_ptr<MuDispatcher> mu_dispatcher_t;

class MuAttentionPool: public MuPool {

private:

    std::vector<std::vector<AttentionBatch>> attn_data_queue;

    AttentionBatch pack_attn_batch(torch::Tensor, metadata_t);

    int tokens_in_layer(int lid) override;

    void process_batch(torch::Tensor tensor, metadata_t &meta) override;

public:

    MuAttentionPool(std::vector<int> layer_ids,
           int device_id,
           std::vector<Channel_t> channels);

    std::vector<AttentionBatch> fetch_largest_batch(int *layer_id = nullptr);

    std::vector<AttentionBatch> fetch_batch_from(int layer_id, int num_batches);

    // for debug use only
    void __set_attn_data_queue(
        std::vector<std::vector<AttentionBatch>> data_queue, 
        std::vector<int> token_per_layer,
        int largest_batch_id) {
        this->attn_data_queue = data_queue;
        this->tokens_per_layer_ = token_per_layer;
        this->largest_batch_layer_id_ = largest_batch_id;
    }
};

typedef std::shared_ptr<MuAttentionPool> mu_attn_pool_t;