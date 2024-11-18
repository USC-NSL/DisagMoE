#pragma once

#include <queue>
#include <vector>
#include <unordered_map>
#include <memory>

#include "datatypes.hpp"

typedef std::shared_ptr<std::vector<int>> block_list_t;

class BlockManager {

private:

    int num_blocks_;

    int reserved_blocks_; // reserved for decoding sequences, not available for prefilling
 
    int block_size_;

    std::queue<int> free_blocks_;

    std::unordered_map<int , block_list_t> block_tables_{};

public:

    BlockManager(int block_size, int num_blocks, int reserved_blocks);

    bool can_allocate(int seq_len);

    void release(int seq_ids);

    void batch_release(const std::vector<int> &seq_ids);

    void allocate(int seq_id, int seq_len);

    void free(int seq_id);

    bool can_append();

    void append_block(int seq_id);

    int num_free_blocks();

    block_list_t get_seq_block_list(int seq_id);

    bool has_seq_block_list(int seq_id);

    void append_tokens(int seq_id, int context_len, int num_tokens);

    void update_block_table(attn_metadata_t meta, const std::vector<int> &decode_seq_lens);

    std::pair<std::vector<int>, std::vector<int>> prepare_block_table(attn_metadata_t meta, const std::vector<int> &decode_seq_lens);
};

typedef std::shared_ptr<BlockManager> block_manager_t;

typedef std::vector<block_list_t> block_table_t;