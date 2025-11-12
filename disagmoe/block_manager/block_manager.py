# @copyright SGLang
# copied and adapted from sglang/srt/mem_cache/mem_pool.py and sglang/srt/mem_cache/allocator.py
import torch
import triton
import triton.language as tl
import numpy as np
from typing import List
from disagmoe.utils.utils import nvtx_range
from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.frontend.datatypes import AttentionForwardBatch
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from disagmoe.block_manager.mem_pool import ReqToTokenPool, TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator

from disagmoe_c import BlockManager as BlockManager_C, BatchMetadata as BatchMetadata_C, rebind_batch_info_tensor

class BaseBlockManager:
    """Base class for block managers"""
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, max_running_reqs: int, device: str = "cuda"):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.max_running_reqs = max_running_reqs
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.decode_seq_lens = {}  # Track sequence lengths for each request
        self.query_start_loc_cuda_buffer = torch.arange(self.model_config.max_batch_size_attn + 1, dtype=torch.int32, device=self.device)
        
    def reset_state(self):
        pass
    
    def update_block_table(self, meta_c: BatchMetadata_C, batch: AttentionForwardBatch):
        pass
    
    def pack_flash_attn_metadata(self, meta_c: BatchMetadata_C, batch: AttentionForwardBatch, dummy_cache: bool = False) -> FlashAttentionMetadata:
        pass
    
    def release_seqs(self, req_ids: List[int]):
        pass
    

GPU_PAGE_SIZE = 1 << 16

def get_cuda_aligned_tensor(numel: int, dtype, alignment: int = GPU_PAGE_SIZE):
    """
    Allocate a CUDA tensor with a 64KB-aligned data pointer.
    Returns (aligned_tensor, base_tensor).
    
    Args:
        numel (int): Number of elements (not bytes).
        dtype (torch.dtype): Tensor dtype (e.g. torch.int32, torch.int64, torch.float32).
        alignment (int): Alignment in bytes (default 64KB).
    """
    # Element size in bytes
    elem_size = torch.tensor([], dtype=dtype).element_size()
    size_bytes = numel * elem_size

    # 1. Overallocate to ensure we have alignment slack
    buf = torch.empty(size_bytes + alignment, dtype=torch.uint8, device="cuda")

    # 2. Compute aligned start address
    base_addr = buf.data_ptr()
    aligned_addr = (base_addr + alignment - 1) & ~(alignment - 1)
    offset = aligned_addr - base_addr

    # 3. Slice the overallocated buffer to create an aligned view
    aligned_buf = buf[offset:offset + size_bytes]

    # 4. Reinterpret as the requested dtype
    aligned_tensor = aligned_buf.view(dtype)
    
    # 5. Sanity check
    assert aligned_tensor.data_ptr() % alignment == 0, "Alignment failed!"

    return aligned_tensor

class CPUBlockManager(BaseBlockManager):
    """CPU-based block manager for comparison - follows original implementation"""
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, max_running_reqs: int, device: str = "cuda"):
        super().__init__(model_config, cache_config, max_running_reqs, device)
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self._block_mgr = BlockManager_C(self.block_size, self.num_gpu_blocks, 0)
        
        self.use_gdr_copy = True
        self.use_rebind = True
        
        max_forward_batch_size = 256
        max_pages_per_req = self.model_config.max_seq_len // self.cache_config.block_size

        self.block_table_cuda_buffer = get_cuda_aligned_tensor(max_forward_batch_size * max_pages_per_req, torch.int32)
        self.slot_mapping_cuda_buffer = get_cuda_aligned_tensor(max_forward_batch_size, torch.long)
        
        self.seq_lens_cuda_buffer = get_cuda_aligned_tensor(max_forward_batch_size, torch.int32)
        self.context_lens_cuda_buffer = get_cuda_aligned_tensor(max_forward_batch_size, torch.int32)
        self.seq_start_loc_cuda_buffer = get_cuda_aligned_tensor(max_forward_batch_size + 1, torch.int32)
        
        if self.use_gdr_copy:
            self._block_mgr.register_gdr_context(self.block_table_cuda_buffer, self.slot_mapping_cuda_buffer)
            self._block_mgr.register_seq_info_gdr(self.seq_lens_cuda_buffer, self.context_lens_cuda_buffer, self.seq_start_loc_cuda_buffer)
        
        self.block_table_view = torch.empty(0, dtype=torch.int32, device=self.device)
        self.slot_mapping_view = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_view = torch.empty(0, dtype=torch.int32, device=self.device)
        self.context_lens_view = torch.empty(0, dtype=torch.int32, device=self.device)
        self.seq_start_loc_view = torch.empty(0, dtype=torch.int32, device=self.device)
        self.query_start_loc_view = torch.empty(0, dtype=torch.int32, device=self.device)

    def reset_state(self):
        self.release_seqs(list(self.decode_seq_lens.keys()))
        self.decode_seq_lens = {}
    
    def release_seqs(self, req_ids: List[int]):
        req_ids = [req_id for req_id in req_ids if req_id in self.decode_seq_lens]
        self._block_mgr.batch_release(req_ids)
        for req_id in req_ids:
            self.decode_seq_lens.pop(req_id)
    
    @nvtx_range("CPUBlockManager.update_block_table")
    def update_block_table(self, meta_c: BatchMetadata_C, batch: AttentionForwardBatch):
        init_req_ids = batch.req_ids[:batch.num_prefill_seqs]
        decode_req_ids = batch.req_ids
        
        # If the first layer in this attention worker, update block table and decode_seq_lens
        if batch.layer_id == self.model_config.layer_ids[0]:
            # Allocate kv blocks for init seqs, update for all decoding seqs
            for i, req_id in enumerate(init_req_ids):
                self.decode_seq_lens[req_id] = batch.init_prefill_lens[i]
            
            decode_seq_lens = [self.decode_seq_lens.get(req_id) for req_id in decode_req_ids]
            
            # Update block table
            self._block_mgr.update_block_table(meta_c, decode_seq_lens)
            
            # Increment sequence lengths for all sequences
            for i, req_id in enumerate(decode_req_ids):
                decode_seq_lens[i] += 1
                self.decode_seq_lens[req_id] += 1
        else:
            decode_seq_lens = [self.decode_seq_lens.get(req_id) for req_id in decode_req_ids]
            
        batch.seq_lens = decode_seq_lens
        
    @nvtx_range("CPUBlockManager.pack_flash_attn_metadata")
    def pack_flash_attn_metadata(self, meta_c: BatchMetadata_C, batch: AttentionForwardBatch, dummy_cache: bool = False) -> FlashAttentionMetadata:
        if self.use_rebind and self.use_gdr_copy and not dummy_cache:
            return self.pack_flash_attn_metadata_opt(meta_c, batch)
        else:
            return self.pack_flash_attn_metadata_naive(meta_c, batch, dummy_cache)
    
    @nvtx_range("CPUBlockManager.pack_flash_attn_metadata_opt")
    def pack_flash_attn_metadata_opt(
        self, 
        meta_c: BatchMetadata_C, 
        batch: AttentionForwardBatch
    ) -> FlashAttentionMetadata:
        """Pack FlashAttention metadata using CPU approach - follows original implementation"""
        num_tokens = batch.num_decode_tokens + batch.num_prefill_tokens
        num_seqs = batch.num_prefill_seqs + batch.num_decode_tokens
        
        num_pages = self._block_mgr.prepare_block_table_gdr(meta_c, batch.seq_lens)
        self._block_mgr.prepare_seq_info_gdr(meta_c, batch.seq_lens)
        
        rebind_batch_info_tensor(
            num_tokens,
            num_pages,
            self.block_table_view,
            self.slot_mapping_view,
            self.seq_lens_view,
            self.context_lens_view,
            self.seq_start_loc_view,
            self.query_start_loc_view,
            self.block_table_cuda_buffer,
            self.slot_mapping_cuda_buffer,
            self.seq_lens_cuda_buffer,
            self.context_lens_cuda_buffer,
            self.seq_start_loc_cuda_buffer,
            self.query_start_loc_cuda_buffer
        )
        
        block_table_cuda = self.block_table_view
        slot_mapping_cuda = self.slot_mapping_view
        seq_lens_cuda = self.seq_lens_view
        context_lens_cuda = self.context_lens_view
        seq_start_loc_cuda = self.seq_start_loc_view
        query_start_loc = self.query_start_loc_view

        max_decode_seq_len = max(batch.seq_lens) if len(batch.seq_lens) > 0 else 0
        
        batch.seq_lens_tensor = seq_lens_cuda
        
        return FlashAttentionMetadata(
            0,
            0,
            num_tokens,
            slot_mapping_cuda,
            seq_lens=batch.seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            max_decode_query_len=1,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
            block_tables=block_table_cuda,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
        )
        
    @nvtx_range("CPUBlockManager.pack_flash_attn_metadata_naive")
    def pack_flash_attn_metadata_naive(
        self, 
        meta_c: BatchMetadata_C, 
        batch: AttentionForwardBatch, 
        dummy_cache: bool = False
    ) -> FlashAttentionMetadata:
        """Pack FlashAttention metadata using CPU approach - follows original implementation"""
        num_tokens = batch.num_decode_tokens + batch.num_prefill_tokens
        num_seqs = batch.num_prefill_seqs + batch.num_decode_tokens
        
        # 1. prepare block table
        if dummy_cache:
            # dummy_cache is True when _warmup_attn
            block_table_cuda = torch.zeros(
                (num_tokens, num_seqs * self.model_config.max_seq_len // self.block_size), 
                dtype=torch.int32, device=self.device
            )
            slot_mapping_cuda = torch.zeros(
                (num_tokens, ), 
                dtype=torch.int64, device=self.device
            )
        else:
            if self.use_gdr_copy:
                num_pages = self._block_mgr.prepare_block_table_gdr(meta_c, batch.seq_lens)
                block_table_cuda = self.block_table_cuda_buffer[ : num_pages].view(num_tokens, -1)
                slot_mapping_cuda = self.slot_mapping_cuda_buffer[ : num_tokens]
            else:
                block_table_1d = self._block_mgr.prepare_block_table(meta_c, batch.seq_lens)
                slot_mapping_cuda = block_table_1d[-num_tokens:].to(torch.int64)
                block_table_cuda = block_table_1d[:-num_tokens].view(num_tokens, -1)

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        if self.use_gdr_copy:
            self._block_mgr.prepare_seq_info_gdr(meta_c, batch.seq_lens)
            seq_lens_cuda = self.seq_lens_cuda_buffer[:num_seqs]
            context_lens_cuda = self.context_lens_cuda_buffer[:num_seqs]
            seq_start_loc_cuda = self.seq_start_loc_cuda_buffer[:num_seqs + 1]
        else:
            batch_infos_cuda = self._block_mgr.prepare_seq_info(meta_c, batch.seq_lens)
            seq_lens_cuda = batch_infos_cuda[ : num_seqs]
            context_lens_cuda = batch_infos_cuda[num_seqs : num_seqs + num_seqs]
            seq_start_loc_cuda = batch_infos_cuda[num_seqs + num_seqs : ]
                
        query_start_loc = self.query_start_loc_cuda_buffer[ : num_tokens + 1]
        max_decode_seq_len = max(batch.seq_lens) if len(batch.seq_lens) > 0 else 0
        
        batch.seq_lens_tensor = seq_lens_cuda
        
        return FlashAttentionMetadata(
            0,
            0,
            num_tokens,
            slot_mapping_cuda,
            seq_lens=batch.seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            max_decode_query_len=1,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
            block_tables=block_table_cuda,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
        )


class GPUBlockManager(BaseBlockManager):
    """GPU-based block manager for comparison"""
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, max_running_reqs: int, device: str = "cuda"):
        super().__init__(model_config, cache_config, max_running_reqs, device)
        
        # Initialize GPU-based components
        self.req_to_indice = {}
        self.decode_seq_lens = {}
        self.req_seq_lens = torch.empty(self.max_running_reqs, dtype=torch.int32, device=self.device)
        
        # Initialize token allocator and req pool
        self.max_running_reqs = max_running_reqs
        self.req_to_token_pool = ReqToTokenPool(
            self.max_running_reqs,
            model_config.max_seq_len,
            device,
        )
        
        if self.cache_config.block_size == 1:
            self.token_allocator = TokenToKVPoolAllocator(
                self.num_gpu_blocks,
                self.model_config.dtype,
                self.device,
                need_sort=False,
            )
        else:
            assert False, "Paged allocator is not supported yet"
            self.token_allocator = PagedTokenToKVPoolAllocator(
                self.num_gpu_blocks,
                self.cache_config.block_size,
                self.model_config.dtype,
                self.device,
                need_sort=False,    
            )
        
        self.seq_start_loc = torch.zeros(self.model_config.max_batch_size_attn + 1, dtype=torch.int32, device=self.device)
    
    def reset_state(self):
        self.decode_seq_lens = {}
        self.req_to_indice = {}
        self.req_seq_lens.zero_()
        self.req_to_token_pool.clear()
        self.token_allocator.clear()  # Reset the token allocator
        
    def release_seqs(self, req_ids: List[int]):
        req_indices = [self.req_to_indice.get(i) for i in req_ids if i in self.decode_seq_lens]
        req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device=self.device)
        self.req_to_token_pool.free(req_indices)
        # get_logger().info(f"releasing seqs {req_ids}")

        self.token_allocator.free_group_begin()
        for req_id, req_indice in zip(req_ids, req_indices):
            # NOTE: single read/write to python dict is thread-safe due to GIL, but iterating should be protected by a lock
            seq_len = self.decode_seq_lens[req_id]
            kv_indices = self.req_to_token_pool.req_to_token[req_indice, : seq_len]
            self.token_allocator.free(kv_indices)
            self.decode_seq_lens.pop(req_id)
        self.token_allocator.free_group_end()
    
    @nvtx_range("GPUBlockManager.update_block_table")
    def update_block_table(self, meta_c: BatchMetadata_C, batch: AttentionForwardBatch):
        init_req_ids = batch.req_ids[:batch.num_prefill_seqs]
        running_seq_ids = batch.req_ids[batch.num_prefill_seqs:]
        req_ids = batch.req_ids
        num_tokens = batch.num_decode_tokens + batch.num_prefill_tokens
        
        if batch.layer_id == 0:  # First layer
            # Allocate prefill token slots for init seqs
            new_req_indices = self.req_to_token_pool.alloc(batch.num_prefill_seqs)
            for i, req_id in enumerate(init_req_ids):
                req_indice = new_req_indices[i]
                self.req_to_indice[req_id] = req_indice
                prefill_kv_locs = self.token_allocator.alloc(batch.init_prefill_lens[i])
                self.req_to_token_pool.write((req_indice, slice(0, batch.init_prefill_lens[i])), prefill_kv_locs)
                self.decode_seq_lens[req_id] = batch.init_prefill_lens[i]
                
            running_req_indices = [self.req_to_indice.get(req_id) for req_id in running_seq_ids]
            batch_req_indices = running_req_indices + new_req_indices
            batch_req_indices_tensor = torch.tensor(batch_req_indices, dtype=torch.int32, device=self.device)
            
            seq_lens = [self.decode_seq_lens.get(req_id) for req_id in req_ids]
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
            
            for i, req_id in enumerate(req_ids):
                seq_lens[i] += 1
                self.decode_seq_lens[req_id] += 1
                
            increment_locs = self.token_allocator.alloc(num_tokens)
            self.req_to_token_pool.write_loc(batch_req_indices_tensor, seq_lens_tensor, increment_locs.to(torch.int32))
            seq_lens_tensor = seq_lens_tensor + 1
            self.req_seq_lens[batch_req_indices_tensor] = seq_lens_tensor
        else:
            seq_lens = [self.decode_seq_lens.get(req_id) for req_id in req_ids]
            batch_req_indices = [self.req_to_indice.get(req_id) for req_id in req_ids]
            batch_req_indices_tensor = torch.tensor(batch_req_indices, dtype=torch.int32, device=self.device)
            seq_lens_tensor = self.req_seq_lens[batch_req_indices_tensor]
            
        batch.seq_lens = seq_lens
        batch.seq_lens_tensor = seq_lens_tensor
        batch.req_indices = batch_req_indices
        batch.req_indices_tensor = batch_req_indices_tensor
        
        return seq_lens
    
    @nvtx_range("GPUBlockManager.pack_flash_attn_metadata")
    def pack_flash_attn_metadata(
            self, 
            meta_c: BatchMetadata_C, 
            batch: AttentionForwardBatch, 
            dummy_cache: bool = False
        ) -> FlashAttentionMetadata:
        """Pack FlashAttention metadata using GPU approach"""
        num_tokens = batch.num_decode_tokens + batch.num_prefill_tokens
        num_seqs = batch.num_prefill_seqs + batch.num_decode_tokens

        seq_lens_cuda = batch.seq_lens_tensor
        context_lens_cuda = seq_lens_cuda - 1
        torch.cumsum(seq_lens_cuda, dim=0, out=self.seq_start_loc[1 : num_tokens + 1])
        seq_start_loc_cuda = self.seq_start_loc[ : num_tokens + 1]
        query_start_loc = self.query_start_loc_cuda_buffer[ : num_tokens + 1]
        seq_lens = batch.seq_lens
        max_decode_seq_len = max(seq_lens) if len(seq_lens) > 0 else 0
        
        if dummy_cache:
            block_table_cuda = torch.arange(
                num_seqs * max_decode_seq_len // self.cache_config.block_size, 
                dtype=torch.int32, device=self.device
            ).view(num_seqs, -1)
            slot_mapping_cuda = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        else:
            req_indices = batch.req_indices_tensor
            block_table_cuda = self.req_to_token_pool.get_block_table(req_indices, max_decode_seq_len)
            slot_mapping_cuda = self.req_to_token_pool.req_to_token[req_indices, context_lens_cuda].to(torch.int64)

        return FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_tokens,
            slot_mapping=slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=1,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            max_decode_query_len=1,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
            block_tables=block_table_cuda,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )
    