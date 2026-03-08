#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <condition_variable>
#include <set>
#include <unordered_map>
#include <memory>

#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
#include "comm.h"
#include "transport_factory.h"
#include "layer.h"

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

    virtual ~MuHelper();

    void start();

    void init_cuda_device();

    virtual void terminate();

    int get_device_id();

};


class MuDispatcher: public MuHelper {
    
protected:
    char device_id_str[16];

    std::queue<std::pair<TokenBatch, int>> send_queue;
    std::mutex mtx;
    std::condition_variable cv;

    std::vector<disagmoe::MqSocketPtr> peer_mq;

    std::queue<std::pair<TokenBatch, cudaEvent_t>> pending_sends;

    ParallelConfig cfg;

    virtual void _send_once(TokenBatch batch) = 0;

    void _send_batch(int cid, uintptr_t buf, const BatchMetadata& meta);

    void clean_pending_sends();

    void send_batch_nonblocking(int cid, const TokenBatch &batch);

    void run() override;


public:
    MuDispatcher(std::vector<int> layer_ids, 
                 int device_id, 
                 ParallelConfig cfg, 
                 std::vector<Channel_t> channels);

    void put(TokenBatch batch, int rank = 0);

};


class MuAttnDispatcher: public MuDispatcher {

protected:
    std::vector<int> exp_channels;
    int max_exp_id;

    std::vector<std::vector<int>> _inner_expert_ranks;

    void _send_once(TokenBatch batch) override;

    int _encode(int exp_layer_id, int exp_id) const;

    int _get_rank(int exp_layer_id, int exp_id) const;

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
    std::vector<std::vector<int>> attn_channel;

    void _send_once(TokenBatch batch) override;
    virtual int _get_attn_channel(int layer_id, int dp_rank);

public:
    MuExpertDispatcher(std::vector<int> layer_ids, 
                       int device_id, 
                       ParallelConfig cfg,
                       std::vector<Channel_t> channels={},
                       std::vector<ChannelInfo> channel_infos={});
    
    void debug_put(TokenBatch batch);
};

class MuPool: public MuHelper {
protected:
    bool is_attn;
    std::vector<Channel_t> peer_channels;

    disagmoe::MqSocketPtr mq;

    int num_layers;
    int num_groups;

    std::vector<std::vector<TokenBatch>> data_queue;
    std::vector<int> layer_id_P2V; // physical layer id to virtual layer id (within this worker)
    std::vector<int> layer_id_V2P; // virtual layer id (within this worker) to physical layer id

    std::mutex request_mutex;
    std::mutex batch_mutex;

    std::condition_variable request_cv;
    int cur_request_count{0};

    int largest_batch_size_{0};
    int largest_batch_layer_id_{-1};
    std::vector<int> tokens_per_layer_;
    std::vector<int> num_batches_per_layer_;

    std::mutex timer_mutex;
    std::map<int, clock_t> queueing_timers;

    std::shared_ptr<LayerSchedulerBase> layer_scheduler;

    void recv_metadata(int &peer_id, batch_metadata_t &meta, bool non_blocking = false);

    virtual void process_batch(torch::Tensor tensor, batch_metadata_t &meta) = 0;

    void start_queueing_timer(const std::vector<int> &req_ids);

    inline int get_layer_group_id(int layer_id, int group_id) {
        return layer_id * num_groups + group_id;
    }

    struct MuPoolPendingRecv {
        int peer_id;
        batch_metadata_t meta;
        torch::Tensor tensor;
        cudaEvent_t event;  // For async NCCL recv completion polling
    };

public:
    MuPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels,
        int num_groups = 1
    );

    virtual ~MuPool();

    void run() override;

    int get_num_layers() { return num_layers; }

    int get_num_groups() { return num_groups; }

    /* 

    for attention, consider waiting sequences,

    1.first layer consider add waiting seqs, count(can_alloc())

    2. later layers pick largest running batch, use token number

    */

    void maintain_largest_batch();

    int get_largest_batch_layer_id() {
        return largest_batch_layer_id_;
    }

    virtual std::vector<int> get_pool_snapshot();

    virtual int tokens_in_layer(int lid);

    virtual int num_batches_in_layer(int lid);

    // Pools do not schedule; the external Scheduler selects layer ids.

    // void set_layer_schedule_type(std::string type);

    // void set_scheduler_block(int step);

    // return average queueing delay    
    float remove_queueing_timer(const std::vector<int> &req_ids);

    void put_batch(TokenBatch batch);

    // Allow external owner (Scheduler) to share/manage layer-wise scheduler state
    void set_layer_scheduler(std::shared_ptr<LayerSchedulerBase> scheduler) { this->layer_scheduler = scheduler; }
    std::shared_ptr<LayerSchedulerBase> get_layer_scheduler() { return this->layer_scheduler; }

    virtual TokenBatch get_batch_from_layer(int layer_id) = 0;
};

class MuExpertPool: public MuPool {
protected:
    std::vector<std::vector<TokenBatch>> data_queue;

    void process_batch(torch::Tensor tensor, batch_metadata_t &meta) override;

public:
    MuExpertPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels,
        int num_groups = 1
    );

    TokenBatch get_batch_from_layer(int layer_id) override;
};



class MuAttentionPool: public MuPool {

private:

    void process_batch(torch::Tensor tensor, batch_metadata_t &meta) override;

protected:

    std::vector<std::vector<TokenBatch>> attn_data_queue;

    TokenBatch pack_attn_batch(torch::Tensor tensor, batch_metadata_t meta);

public:

    MuAttentionPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels
    );

    void put_batch_to_attn_queue(int layer_id, const TokenBatch &batch);

    TokenBatch get_batch_from_layer(int layer_id) override;
};


class TokenTopKPool {

    int top_k;

    std::unordered_map<int, TokenTopKInfo> pool_{}; // mapping from seq_id to corresponding TokenTopKInfo

    std::vector<TokenTopKInfo> ready_tokens{};

public:

    TokenTopKPool(int top_k): top_k(top_k) {}

    void put_batch(TokenBatch batch);

    std::vector<TokenTopKInfo> fetch_ready_tokens();

    int get_top_k() { return top_k; }

    int get_pool_size() { return this->pool_.size(); }

};

class MuAttentionTopKPool: public MuAttentionPool {

private:

    int top_k;

    std::vector<std::vector<TokenTopKInfo>> attn_token_queues;

    std::vector<TokenTopKPool> topk_pools;


    std::vector<TokenTopKInfo> schedule_with_limit();

    void process_batch(torch::Tensor tensor, batch_metadata_t &meta) override;

public:

    MuAttentionTopKPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels,
        int top_k = 1
    );

    int tokens_in_layer(int lid) override;

    TokenBatch get_batch_from_layer(int layer_id) override;

};

typedef std::shared_ptr<MuPool> mu_pool_t;  // For backward compatibility
typedef std::shared_ptr<MuDispatcher> mu_dispatcher_t;

typedef std::shared_ptr<MuExpertPool> mu_expert_pool_t;
typedef std::shared_ptr<MuExpertDispatcher> mu_expert_dispatcher_t;

typedef std::shared_ptr<MuAttentionPool> mu_attn_pool_t;
typedef std::shared_ptr<MuAttnDispatcher> mu_attn_dispatcher_t;

typedef std::shared_ptr<MuAttentionTopKPool> mu_attn_topk_pool_t;
