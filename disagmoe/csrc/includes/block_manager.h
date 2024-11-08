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

    BlockManager(const int &block_size, const int &num_blocks, const int &reserved_blocks);

    bool can_allocate(const int &seq_len);

    void allocate(const int &seq_id, const int &seq_len);

    void free(const int &seq_id);

    bool can_append();

    void append_block(const int& seq_id);

    int num_free_blocks();

    block_list_t get_seq_block_list(const int& seq_id);

    bool has_seq_block_list(const int &seq_id);

    void append_tokens(int seq_id, int context_len, int num_tokens);

    std::vector<std::vector<int>> prepare_block_table(attn_metadata_t meta);
};

typedef std::shared_ptr<BlockManager> block_manager_t;

typedef std::vector<block_list_t> block_table_t;