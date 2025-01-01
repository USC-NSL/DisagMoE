#include "block_manager.h"
#include "logging.h"

#include <vector>
#include <queue>
#include <memory>
#include <mutex>

#include "cuda_utils.h"

BlockManager::BlockManager(int block_size, int num_blocks, int reserved_blocks) {
    num_blocks_ = num_blocks;
    reserved_blocks_ = reserved_blocks;
    block_size_ = block_size;
    
    for (int i = 0; i < num_blocks_ + reserved_blocks_; i++) {
        free_blocks_.push(i);
    }
}

int BlockManager::get_one_free_block() {
    std::lock_guard<std::mutex> lock(free_blocks_lock_);
    ASSERT (free_blocks_.size() > 0);
    int block_id = free_blocks_.front();
    free_blocks_.pop();
    // DMOE_LOG(INFO) << "get_one_free_block, remaining blocks: " << free_blocks_.size() << LEND;
    return block_id;
}

int BlockManager::num_free_blocks() {
    std::lock_guard<std::mutex> lock(free_blocks_lock_);
    return free_blocks_.size();
}

void BlockManager::release(int seq_ids) {
    std::lock_guard<std::mutex> lock(free_blocks_lock_);
    ASSERT (block_tables_.find(seq_ids) != block_tables_.end());
    for (auto &x: (*block_tables_[seq_ids])) {
        free_blocks_.push(x);
    }
    block_tables_.erase(seq_ids);
}

bool BlockManager::can_allocate(int seq_len) {
    int blocks_needed = (seq_len - 1) / block_size_ + 1;
    return num_free_blocks() >= blocks_needed + reserved_blocks_;
}

void BlockManager::batch_release(const std::vector<int> &seq_ids) {
    for (auto &seq_id: seq_ids) {
        release(seq_id);
    }
}

void BlockManager::allocate(int seq_id, int seq_len) {
    AUTO_TX_RANGE;
    // DMOE_LOG(DEBUG) << "allocating for " << seq_id << " " << seq_len << LEND;
    ASSERT (block_tables_.find(seq_id) == block_tables_.end());
    int blocks_needed = (seq_len - 1) / block_size_ + 1;
    
    // DMOE_LOG(INFO) << "blocks_needed = " << blocks_needed << LEND;

    ASSERT (num_free_blocks() >= blocks_needed + reserved_blocks_);
    block_list_t block_list = std::make_shared<std::vector<int>>(std::vector<int>(blocks_needed));
    for (int i = 0; i < blocks_needed; i++) {
        int new_block_id = get_one_free_block();
        (*block_list)[i] = new_block_id;
    }
    block_tables_[seq_id] = block_list;

    // DMOE_LOG(DEBUG) << "allocated for " << seq_id << " " << seq_len << LEND;
}

void BlockManager::append_block(int seq_id) {
    ASSERT (num_free_blocks() > 0);

    int new_block_id = get_one_free_block();

    auto seq_block_list = block_tables_.find(seq_id);
    ASSERT (seq_block_list != block_tables_.end());
    seq_block_list->second->emplace_back(new_block_id);
}

bool BlockManager::can_append() {
    return num_free_blocks() > 0;
}

bool BlockManager::has_seq_block_list(int seq_id) {
    return block_tables_.find(seq_id) != block_tables_.end();
}

block_list_t BlockManager::get_seq_block_list(int seq_id) {
    return block_tables_[seq_id];
}

void BlockManager::append_tokens(int seq_id, int context_len, int num_tokens) {
    tx_range _{"BlockManager::append_tokens"};
    ASSERT (num_tokens >= 1);
    ASSERT (has_seq_block_list(seq_id));
    ASSERT (context_len > 0);

    int remain_slots = block_size_ - context_len % block_size_; 
    if (remain_slots == block_size_) { 
        remain_slots = 0;
    }
    if (num_tokens > remain_slots) {
        int blocks_to_add = (num_tokens - remain_slots - 1) / block_size_ + 1;
        ASSERT (num_free_blocks() > blocks_to_add);
        auto seq_block_list = block_tables_.find(seq_id);
        // DMOE_LOG(INFO) << "append_tokens for sequence: " << seq_id << ", current block_num: " << seq_block_list->second->size() << ", blocks_to_add: " << blocks_to_add << LEND;
        while (blocks_to_add > 0) {
            int block_to_append = get_one_free_block();
            seq_block_list->second->emplace_back(block_to_append);
            blocks_to_add --;
        }
    }
}

void BlockManager::update_block_table(attn_metadata_t meta, const std::vector<int> &context_lens) {
    int num_prefill_seqs = meta->num_prefill_seqs;
    int num_decode_tokens = meta->num_decode_tokens;
    for (int i = 0; i < num_prefill_seqs; i++) {
        int seq_id = meta->seq_ids[i];
        if (!has_seq_block_list(seq_id)) {
            allocate(seq_id, meta->prefill_seq_len[i]);
        } else {
            append_tokens(seq_id, meta->prefill_seq_len[i] - meta->prefill_query_len[i], meta->prefill_query_len[i]);
        }
    }
    for (int i = 0; i < num_decode_tokens; i++) {
        int seq_id = meta->seq_ids[num_prefill_seqs + i];
        ASSERT (has_seq_block_list(seq_id));
        int context_len = context_lens[i];
        append_tokens(seq_id, context_len, 1);
    }
}


torch::Tensor BlockManager::prepare_block_table(attn_metadata_t meta, const std::vector<int> &decode_seq_lens) {
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
    std::vector<int> block_table_1d(n * m + n, -1);
    for (int i = 0; i < n; i++) {
        int id = meta->seq_ids[i];
        block_list_t list = get_seq_block_list(id);
        for (int j = 0; j < list->size(); j++) {
            block_table_1d[i * m + j] = (*list)[j];
        }
    }

    int tokens_in_batch = meta->num_prefill_tokens + meta->num_decode_tokens;

    int slot_idx = n * m;
    for (int i = 0; i < meta->num_prefill_seqs; i++) {
        int q_len = meta->prefill_query_len[i];
        int seq_len = meta->prefill_seq_len[i];
        for (int idx = seq_len - q_len; idx < seq_len; idx++) {
            int block_id = idx / block_size_;
            int id_in_block = idx % block_size_;
            block_table_1d[slot_idx] = block_table_1d[i * m + block_id] * block_size_ + id_in_block;
            slot_idx ++;
        }
    }
    for (int i = meta->num_prefill_tokens; i < tokens_in_batch; i++) {
        int last_idx = decode_seq_lens[i - meta->num_prefill_tokens] - 1; // decode_index should be decode_lens - 1
        int block_id = last_idx / block_size_;
        int id_in_block = last_idx % block_size_;
        block_table_1d[slot_idx] = block_table_1d[i * m + block_id] * block_size_ + id_in_block;
        slot_idx ++;
    }

    return torch::tensor(block_table_1d, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
}