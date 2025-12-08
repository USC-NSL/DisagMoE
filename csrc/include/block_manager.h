#pragma once

#include <queue>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <optional>

#include "datatypes.hpp"
#include "metadata.hpp"
#include "gdr_context.hpp"

typedef std::shared_ptr<std::vector<int>> block_list_t;

class BlockManager {

private:

    int num_blocks_;

    int reserved_blocks_; // reserved for decoding sequences, not available for prefilling
 
    int block_size_;

    std::mutex free_blocks_lock_;

    std::queue<int> free_blocks_{};

    std::unordered_map<int , block_list_t> block_tables_{};

    int get_one_free_block(); 

public:

    BlockManager(int block_size, int num_blocks, int reserved_blocks);

    ~BlockManager();

    bool can_allocate(int seq_len);

    void close();

    void release(int seq_ids);

    void batch_release(const std::vector<int> &seq_ids);

    void allocate(int seq_id, int seq_len);

    bool can_append();

    void append_block(int seq_id);

    int num_free_blocks();

    block_list_t get_seq_block_list(int seq_id);

    bool has_seq_block_list(int seq_id);

    void append_tokens(int seq_id, int context_len, int num_tokens);

    void update_block_table(batch_metadata_t meta, const std::vector<int> &context_lens);

    torch::Tensor prepare_block_table(batch_metadata_t meta, const std::vector<int> &decode_seq_lens);

    int prepare_block_table_gdr(batch_metadata_t meta, const std::vector<int> &decode_seq_lens, GdrContext &block_table_gdr, GdrContext &slot_mapping_gdr);

    // this function is not related to block manager, but we just put it here for convenience
    torch::Tensor prepare_seq_info(batch_metadata_t meta, const std::vector<int> &decode_seq_lens);

    void prepare_seq_info_gdr(batch_metadata_t meta, const std::vector<int> &decode_seq_lens, GdrContext &seq_lens_gdr, GdrContext &context_lens_gdr, GdrContext &seq_start_loc_gdr);
};

typedef std::shared_ptr<BlockManager> block_manager_t;

typedef std::vector<block_list_t> block_table_t;

void rebind_batch_info_tensor(
    int num_tokens,
    int num_pages,
    torch::Tensor &block_table_view,
    torch::Tensor &slot_mapping_view,
    torch::Tensor &seq_lens_view,
    torch::Tensor &context_lens_view,
    torch::Tensor &seq_start_loc_view,
    torch::Tensor &query_start_loc_view,
    const torch::Tensor &block_table_cuda_buffer,
    const torch::Tensor &slot_mapping_cuda_buffer,
    const torch::Tensor &seq_lens_cuda_buffer,
    const torch::Tensor &context_lens_cuda_buffer,
    const torch::Tensor &seq_start_loc_cuda_buffer,
    const torch::Tensor &query_start_loc_cuda_buffer
);

