#pragma once

#include <queue>
#include <vector>
#include <unordered_map>

using block_list_t = std::vector<int>;

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

    block_list_t allocate(const int &seq_id, const int &seq_len);

    void free(const block_list_t& block_list);

    bool can_append();

    block_list_t append_block(const int& seq_id);

    int num_free_blocks();

    block_list_t get_seq_block_list(const int& seq_id);
};