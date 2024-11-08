#include "block_manager.h"
#include "logging.h"

#include <vector>
#include <queue>
#include <memory>

#include "cuda_utils.h"

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

void BlockManager::allocate(const int &seq_id, const int &seq_len) {
    AUTO_TX_RANGE;
    // LOG(DEBUG) << "allocating for " << seq_id << " " << seq_len << LEND;

    ASSERT (block_tables_.find(seq_id) == block_tables_.end());
    int blocks_needed = (seq_len - 1) / block_size_ + 1;
    
    // LOG(INFO) << "blocks_needed = " << blocks_needed << LEND;

    ASSERT (free_blocks_.size() >= blocks_needed + reserved_blocks_);
    block_list_t block_list = std::make_shared<std::vector<int>>(std::vector<int>(blocks_needed));
    for (int i = 0; i < blocks_needed; i++) {
        (*block_list)[i] = free_blocks_.front();
        free_blocks_.pop();
    }
    block_tables_[seq_id] = block_list;

    // LOG(DEBUG) << "allocated for " << seq_id << " " << seq_len << LEND;

}

void BlockManager::free(const int &seq_id) {
    auto it = block_tables_.find(seq_id);
    auto block_list = it->second;
    for (auto &x: (*block_list)) {
        free_blocks_.push(x);
    }
    block_tables_.erase(it);
}

void BlockManager::append_block(const int& seq_id) {
    ASSERT (free_blocks_.size() > 0);
    int block_to_append = free_blocks_.front();
    free_blocks_.pop();

    auto seq_block_list = block_tables_.find(seq_id);
    ASSERT (seq_block_list != block_tables_.end());
    seq_block_list->second->emplace_back(block_to_append);
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

void BlockManager::append_tokens(int seq_id, int context_len, int num_tokens) {
    tx_range _{"BlockManager::append_tokens"};
    ASSERT (num_tokens >= 1);
    ASSERT (has_seq_block_list(seq_id));

    int remain_slots = block_size_ - context_len % block_size_; 
    if (remain_slots == block_size_) { 
        remain_slots = 0;
    }
    // NOTE(hogura|20241015): here use >= instead of >, otherwise no blocks available at block_size_.
    if (num_tokens > remain_slots) {
        int blocks_to_add = (num_tokens - remain_slots - 1) / block_size_ + 1;
        ASSERT (free_blocks_.size() > blocks_to_add);
        auto seq_block_list = block_tables_.find(seq_id);
        while (blocks_to_add > 0) {
            int block_to_append = free_blocks_.front();
            free_blocks_.pop();
            seq_block_list->second->emplace_back(block_to_append);
            blocks_to_add --;
        }
    }
}

std::vector<int> BlockManager::prepare_block_table(attn_metadata_t meta) {
    AUTO_TX_RANGE;
    // It should be ensured that every seq in batch has been alocated cache blocks
    // For simple case, we allocate cache block in this function, which means every sequence is forcely accepted
    int n = meta->seq_ids.size(); // decode seqs are already allocated in previous steps
    size_t m = 0;
    for (int i = 0; i < n; i++) {
        int id = meta->seq_ids[i];
        ASSERT (has_seq_block_list(id));
        block_list_t list = get_seq_block_list(id);
        m = std::max(m, list->size());
    }
    std::vector<int> result(n * m, -1);
    for (int i = 0; i < n; i++) {
        int id = meta->seq_ids[i];
        block_list_t list = get_seq_block_list(id);
        for (int j = 0; j < list->size(); j++) {
            result[i * m + j] = (*list)[j];
        }
    }
    return result;
}