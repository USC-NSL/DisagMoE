#include "block_manager.h"
#include <vector>
#include <queue>
#include <cassert>
#include <memory>

BlockManager::BlockManager(const int &block_size, const int &num_blocks, const int &reserved_blocks) {
    num_blocks_ = num_blocks;
    reserved_blocks_ = reserved_blocks;
    block_size_ = block_size;
    
    for (int i = 0; i < num_blocks_; i++) {
        free_blocks_.push(i);
    }
}

bool BlockManager::can_allocate(const int &seq_len) {
    int blocks_needed = (seq_len - 1) / block_size_ + 1;
    return free_blocks_.size() >= blocks_needed + reserved_blocks_;
}

block_list_t BlockManager::allocate(const int &seq_id, const int &seq_len) {
    assert (block_tables_.find(seq_id) == block_tables_.end());
    int blocks_needed = (seq_len - 1) / block_size_ + 1;
    assert (free_blocks_.size() >= blocks_needed + reserved_blocks);
    block_list_t block_list = std::make_shared<std::vector<int>>(std::vector<int>(blocks_needed));
    for (int i = 0; i < blocks_needed; i++) {
        (*block_list)[i] = free_blocks_.front();
        free_blocks_.pop();
    }
    block_tables_[seq_id] = block_list;
    return block_list;
}

void BlockManager::free(const int &seq_id) {
    auto it = block_tables_.find(seq_id);
    auto block_list = it->second;
    for (auto &x: (*block_list)) {
        free_blocks_.push(x);
    }
    block_tables_.erase(it);
}

block_list_t BlockManager::append_block(const int& seq_id) {
    assert (free_blocks_.size() > 0);
    int block_to_append = free_blocks_.front();
    free_blocks_.pop();

    auto seq_block_list = block_tables_.find(seq_id);
    assert (seq_block_list != block_tables_.end());
    seq_block_list->second->emplace_back(block_to_append);
    return seq_block_list->second;
}

bool BlockManager::can_append() {
    return free_blocks_.size() > 0;
}

int BlockManager::num_free_blocks() {
    return free_blocks_.size();
}

bool BlockManager::has_seq_block_list(const int &seq_id) {
    return block_tables_.find(seq_id) != block_tables_.end();
}

block_list_t BlockManager::get_seq_block_list(const int &seq_id) {
    return block_tables_[seq_id];
}