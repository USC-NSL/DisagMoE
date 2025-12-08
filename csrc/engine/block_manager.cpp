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

BlockManager::~BlockManager() {
    close();
}

void BlockManager::close() { }

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

void BlockManager::release(int req_ids) {
    std::lock_guard<std::mutex> lock(free_blocks_lock_);
    ASSERT (block_tables_.find(req_ids) != block_tables_.end());
    for (auto &x: (*block_tables_[req_ids])) {
        free_blocks_.push(x);
    }
    block_tables_.erase(req_ids);
}

bool BlockManager::can_allocate(int seq_len) {
    int blocks_needed = (seq_len - 1) / block_size_ + 1;
    return num_free_blocks() >= blocks_needed + reserved_blocks_;
}

void BlockManager::batch_release(const std::vector<int> &req_ids) {
    for (auto &req_id: req_ids) {
        release(req_id);
    }
}

void BlockManager::allocate(int req_id, int seq_len) {
    AUTO_TX_RANGE;
    // DMOE_LOG(DEBUG) << "allocating for " << req_id << " " << seq_len << LEND;
    ASSERT (block_tables_.find(req_id) == block_tables_.end());
    int blocks_needed = (seq_len - 1) / block_size_ + 1;
    
    // DMOE_LOG(INFO) << "blocks_needed = " << blocks_needed << LEND;

    ASSERT (num_free_blocks() >= blocks_needed + reserved_blocks_);
    block_list_t block_list = std::make_shared<std::vector<int>>(std::vector<int>(blocks_needed));
    for (int i = 0; i < blocks_needed; i++) {
        int new_block_id = get_one_free_block();
        (*block_list)[i] = new_block_id;
    }
    block_tables_[req_id] = block_list;

    // DMOE_LOG(DEBUG) << "allocated for " << req_id << " " << seq_len << LEND;
}

void BlockManager::append_block(int req_id) {
    ASSERT (num_free_blocks() > 0);

    int new_block_id = get_one_free_block();

    auto seq_block_list = block_tables_.find(req_id);
    ASSERT (seq_block_list != block_tables_.end());
    seq_block_list->second->emplace_back(new_block_id);
}

bool BlockManager::can_append() {
    return num_free_blocks() > 0;
}

bool BlockManager::has_seq_block_list(int req_id) {
    return block_tables_.find(req_id) != block_tables_.end();
}

block_list_t BlockManager::get_seq_block_list(int req_id) {
    return block_tables_[req_id];
}

void BlockManager::append_tokens(int req_id, int context_len, int num_tokens) {
    tx_range _{"BlockManager::append_tokens"};
    ASSERT (num_tokens >= 1);
    ASSERT (has_seq_block_list(req_id));
    ASSERT (context_len > 0);

    int remain_slots = block_size_ - context_len % block_size_; 
    if (remain_slots == block_size_) { 
        remain_slots = 0;
    }
    if (num_tokens > remain_slots) {
        int blocks_to_add = (num_tokens - remain_slots - 1) / block_size_ + 1;
        ASSERT (num_free_blocks() > blocks_to_add);
        auto seq_block_list = block_tables_.find(req_id);
        // DMOE_LOG(INFO) << "append_tokens for sequence: " << req_id << ", current block_num: " << seq_block_list->second->size() << ", blocks_to_add: " << blocks_to_add << LEND;
        while (blocks_to_add > 0) {
            int block_to_append = get_one_free_block();
            seq_block_list->second->emplace_back(block_to_append);
            blocks_to_add --;
        }
    }
}

void BlockManager::update_block_table(batch_metadata_t meta, const std::vector<int> &context_lens) {
    // first allocate cache blocks for init seqs, then append a slot for all seqs
    int num_prefill_tokens = meta->num_prefill_tokens.value();
    int num_decode_tokens = meta->num_decode_tokens.value();
    for (int i = 0; i < num_prefill_tokens; i++) {
        int req_id = meta->req_ids[i];
        ASSERT (!has_seq_block_list(req_id));
        allocate(req_id, meta->init_prefill_lens[i]);
    }
    for (int i = 0; i < meta->num_tokens(); i++) {
        int req_id = meta->req_ids[i];
        ASSERT (has_seq_block_list(req_id));
        int context_len = context_lens[i];
        append_tokens(req_id, context_len, 1);
    }
}


torch::Tensor BlockManager::prepare_block_table(batch_metadata_t meta, const std::vector<int> &decode_seq_lens) {
    AUTO_TX_RANGE;
    // It should be ensured that every seq in batch has been alocated cache blocks
    // For simple case, we allocate cache block in this function, which means every sequence is forcely accepted
    int n = meta->num_tokens(); // decode seqs are already allocated in previous steps
    size_t m = 0;
    for (int i = 0; i < n; i++) {
        int id = meta->req_ids[i];
        ASSERT (has_seq_block_list(id));
        block_list_t list = get_seq_block_list(id);
        m = std::max(m, list->size());
    }
    std::vector<int> block_table_1d(n * m + n, -1);
    for (int i = 0; i < n; i++) {
        int id = meta->req_ids[i];
        block_list_t list = get_seq_block_list(id);
        for (int j = 0; j < list->size(); j++) {
            block_table_1d[i * m + j] = (*list)[j];
        }
    }

    int slot_idx = n * m;
    for (int i = 0; i < n; i++) {
        int last_idx = decode_seq_lens[i] - 1; // decode_index should be decode_lens - 1
        int block_id = last_idx / block_size_;
        int id_in_block = last_idx % block_size_;
        block_table_1d[slot_idx] = block_table_1d[i * m + block_id] * block_size_ + id_in_block;
        slot_idx ++;
    }
    auto block_table_1d_pinned = torch::tensor(block_table_1d, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU, 0).pinned_memory(true));

    return block_table_1d_pinned.to(torch::kCUDA, true);
}

int BlockManager::prepare_block_table_gdr(
    batch_metadata_t meta, 
    const std::vector<int> &decode_seq_lens,
    GdrContext &block_table_gdr,
    GdrContext &slot_mapping_gdr
) {
    int n = meta->num_tokens(); // decode seqs are already allocated in previous steps
    size_t m = 0;
    for (int i = 0; i < n; i++) {
        int id = meta->req_ids[i];
        ASSERT (has_seq_block_list(id));
        block_list_t list = get_seq_block_list(id);
        m = std::max(m, list->size());
    }
    std::vector<int> block_table(n * m, -1);
    for (int i = 0; i < n; i++) {
        int id = meta->req_ids[i];
        block_list_t list = get_seq_block_list(id);
        for (int j = 0; j < list->size(); j++) {
            block_table[i * m + j] = (*list)[j];
        }
    }

    block_table_gdr.copy_from_host(block_table.data(), n * m * sizeof(int));

    std::vector<int64_t> slot_mapping(n);
    for (int i = 0; i < n; i++) {
        int last_idx = decode_seq_lens[i] - 1; // decode_index should be decode_lens - 1
        int block_id = last_idx / block_size_;
        int id_in_block = last_idx % block_size_;
        slot_mapping[i] = static_cast<int64_t>(block_table[i * m + block_id] * block_size_ + id_in_block);
    }
    slot_mapping_gdr.copy_from_host(slot_mapping.data(), n * sizeof(int64_t));

    return m;
}

torch::Tensor BlockManager::prepare_seq_info(batch_metadata_t meta, const std::vector<int> &decode_seq_lens) {
    int num_tokens = meta->num_tokens();
    int num_seqs = num_tokens;

    std::vector<int> batch_infos(num_seqs + num_seqs + (num_seqs + 1), 0);

    std::copy(decode_seq_lens.begin(), decode_seq_lens.end(), batch_infos.begin());
    std::vector<int> seq_lens(batch_infos.begin(), batch_infos.begin() + num_seqs);

    for (int i = 0; i < num_tokens; i++) {
        batch_infos[num_seqs + i] = decode_seq_lens[i] - 1;
    }  

    int base = num_seqs + num_seqs;

    for (int i = 1; i <= num_seqs; i++) {
        batch_infos[base + i] = batch_infos[base + i - 1] + decode_seq_lens[i - 1];
    }

    return torch::tensor(batch_infos, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
}

void BlockManager::prepare_seq_info_gdr(
    batch_metadata_t meta, 
    const std::vector<int> &decode_seq_lens,
    GdrContext &seq_lens_gdr,
    GdrContext &context_lens_gdr,
    GdrContext &seq_start_loc_gdr
) {
    int num_tokens = meta->num_tokens();
    int num_seqs = num_tokens;

    std::vector<int> context_lens(num_seqs);
    for (int i = 0; i < num_seqs; i++) {
        context_lens[i] = decode_seq_lens[i] - 1;
    }

    std::vector<int> seq_start_loc(num_seqs + 1, 0);
    for (int i = 1; i <= num_seqs; i++) {
        seq_start_loc[i] = seq_start_loc[i - 1] + decode_seq_lens[i - 1];
    }

    seq_lens_gdr.copy_from_host(decode_seq_lens.data(), num_seqs * sizeof(int));
    context_lens_gdr.copy_from_host(context_lens.data(), num_seqs * sizeof(int));
    seq_start_loc_gdr.copy_from_host(seq_start_loc.data(), (num_seqs + 1) * sizeof(int));
}

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
) {
    rebind_2d_tensor(block_table_view, block_table_cuda_buffer, 0, num_tokens, num_pages, num_pages, 1);
    rebind_1d_tensor(slot_mapping_view, slot_mapping_cuda_buffer, 0, num_tokens);
    rebind_1d_tensor(seq_lens_view, seq_lens_cuda_buffer, 0, num_tokens);
    rebind_1d_tensor(context_lens_view, context_lens_cuda_buffer, 0, num_tokens);
    rebind_1d_tensor(seq_start_loc_view, seq_start_loc_cuda_buffer, 0, num_tokens + 1);
    rebind_1d_tensor(query_start_loc_view, query_start_loc_cuda_buffer, 0, num_tokens + 1);
}