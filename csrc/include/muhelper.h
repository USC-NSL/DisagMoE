#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <condition_variable>
#include <set>
#include <unordered_map>

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

    virtual void terminate();

    int get_device_id();

};


class MuDispatcher: public MuHelper {
    
protected:
    char device_id_str[3];

    int peer_zmq_port_offset{0};

    std::queue<std::pair<TensorBatch, int>> send_queue;
    std::mutex mtx;
    std::condition_variable cv;

    // ctx must be ahead of mq
    std::vector<zmq::context_t> peer_ctx;
    std::vector<zmq::socket_t> peer_mq;

    // use for nccl group channels
    std::vector<bool> is_group_channels;
    std::vector<std::shared_ptr<NcclGroupChannel>> group_channels;

    ParallelConfig cfg;

    virtual void _send_once(TensorBatch batch) = 0;

    void _send_batch(int cid, uintptr_t buf, const Metadata& meta);

    void run() override;

    bool _is_group_channel(int cid) const;

public:
    MuDispatcher(std::vector<int> layer_ids, 
                 int device_id, 
                 ParallelConfig cfg, 
                 std::vector<Channel_t> channels,
                 const std::vector<bool> &is_group_channels={});

    void put(TensorBatch batch, int rank = 0);
};


class MuAttnDispatcher: public MuDispatcher {

protected:
    std::vector<int> exp_channels;
    int max_exp_id;

    std::vector<std::vector<int>> _inner_expert_ranks;

    void _send_once(TensorBatch batch) override;

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
    int sampler_channel_id;

    void _send_once(TensorBatch batch) override;
    virtual int _get_attn_channel(int req_id, int layer_id);

public:
    MuExpertDispatcher(std::vector<int> layer_ids, 
                       int device_id, 
                       ParallelConfig cfg,
                       std::vector<Channel_t> channels={},
                       std::vector<ChannelInfo> channel_infos={},
                       const std::vector<bool> &is_group_channels={});
    
    void debug_put(TensorBatch batch);
};

enum LayerSchedulePolicy {
    BASE,
    ADVANCED,
    GROUP,
};

class LayerScheduler;

class AdvancedLayerScheduler;

class GroupLayerScheduler;

class MuPool: public MuHelper {
protected:
    bool is_attn;
    std::vector<Channel_t> peer_channels;

    // ctx must be ahead of mq
    zmq::context_t ctx;
    zmq::socket_t mq;

    int num_layers;
    int num_groups;

    std::vector<std::vector<TensorBatch>> data_queue;
    std::vector<int> layer_id_P2V; // physical layer id to virtual layer id (within this worker)
    std::vector<int> layer_id_V2P; // virtual layer id (within this worker) to physical layer id

    std::mutex request_mutex;
    std::mutex batch_mutex;

    std::condition_variable request_cv;
    int cur_request_count{0};

    int largest_batch_size_{0};
    int largest_batch_layer_id_{-1};
    int local_zmq_port_offset{0};
    std::vector<int> tokens_per_layer_;
    std::vector<int> num_batches_per_layer_;

    int max_batch_size;

    std::mutex timer_mutex;
    std::map<int, clock_t> queueing_timers;

    std::shared_ptr<LayerScheduler> layer_scheduler;

    void recv_metadata(int &peer_id, metadata_t &meta);

    void recv_tensor(int peer_id, uintptr_t tensor_buf, metadata_t &meta);

    virtual void process_batch(torch::Tensor tensor, metadata_t &meta, bool send_from_zmq=true);

    void start_queueing_timer(const std::vector<int> &req_ids);

    inline int get_layer_group_id(int layer_id, int group_id) {
        return layer_id * num_groups + group_id;
    }

public:
    MuPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels,
        LayerSchedulePolicy policy = LayerSchedulePolicy::ADVANCED,
        int num_groups = 1,
        bool is_attn = false
    );

    void run() override;

    void wait_for_new_requests();

    void set_max_batch_size(int max_batch_size);

    /* 

    for attention, consider waiting sequences,

    1.first layer consider add waiting seqs, count(can_alloc())

    2. later layers pick largest running batch, use token number

    */
    std::vector<TensorBatch> fetch_largest_batch();

    void maintain_largest_batch();

    int get_largest_batch_layer_id() {
        return largest_batch_layer_id_;
    }

    std::vector<int> get_pool_snapshot();

    virtual int tokens_in_layer(int lid);

    virtual int num_batches_in_layer(int lid);

    int schedule_layer_id();

    // void set_layer_schedule_type(std::string type);

    // void set_scheduler_block(int step);

    // return average queueing delay    
    float remove_queueing_timer(const std::vector<int> &req_ids);
};

typedef std::shared_ptr<MuPool> mu_pool_t;
typedef std::shared_ptr<MuDispatcher> mu_dispatcher_t;

class MuAttentionPool: public MuPool {

private:

    // large device group: [previous_dispatcher; current_driver; current_workers]
    // small device group: [current_driver; current_workers]
    std::thread pool_thread;
    std::vector<std::thread> group_threads;

    std::vector<std::vector<AttentionBatch>> attn_data_queue;

    AttentionBatch pack_attn_batch(torch::Tensor tensor, metadata_t meta);

    void process_batch(torch::Tensor tensor, metadata_t &meta, bool send_from_zmq=true) override;

protected:

    std::vector<int> device_group_ids;
    std::shared_ptr<NcclGroupChannel> group_comm;

public:

    MuAttentionPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels,
        std::vector<int> device_group_ids = {},
        Channel_t group_comm = nullptr,
        LayerSchedulePolicy policy = LayerSchedulePolicy::ADVANCED
    );

    virtual std::vector<AttentionBatch> fetch_largest_batch(int *layer_id = nullptr);

    std::vector<AttentionBatch> fetch_batch_from(int layer_id, std::set<int> &seq_ids);

    void run() override;

    void terminate() override;

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

class TokenTopKPool {

    int top_k;

    std::unordered_map<int, TokenTopKInfo> pool_{}; // mapping from seq_id to corresponding TokenTopKInfo

    std::vector<TokenTopKInfo> ready_tokens{};

public:

    TokenTopKPool(int top_k): top_k(top_k) {}

    void put_batch(TensorBatch batch);

    std::vector<TokenTopKInfo> fetch_ready_tokens();

    int get_top_k() { return top_k; }

};

class MuAttentionTopKPool: public MuAttentionPool {

    int top_k;

    std::vector<std::vector<TokenTopKInfo>> attn_token_queues;

    std::vector<TokenTopKPool> topk_pools;


    std::vector<TokenTopKInfo> schedule_with_limit();

    void process_batch(torch::Tensor tensor, metadata_t &meta, bool send_from_zmq=true) override;

public:

    MuAttentionTopKPool(
        std::vector<int> layer_ids,
        int device_id,
        std::vector<Channel_t> channels,
        std::vector<int> device_group_ids = {},
        Channel_t group_comm = nullptr,
        int top_k = 1,
        LayerSchedulePolicy policy = LayerSchedulePolicy::ADVANCED
    );

    int tokens_in_layer(int lid) override;

    std::vector<AttentionBatch> fetch_largest_batch(int *layer_id = nullptr) override;

};