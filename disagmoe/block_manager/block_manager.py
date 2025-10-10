# @copyright SGLang
# copied and adapted from sglang/srt/mem_cache/mem_pool.py and sglang/srt/mem_cache/allocator.py
import torch
import triton
import triton.language as tl
import numpy as np
from typing import List
from disagmoe.utils.utils import nvtx_range
from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from disagmoe.block_manager.mem_pool import ReqToTokenPool, TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator

from disagmoe_c import BlockManager as BlockManager_C, AttentionBatchMetadata as AttentionBatchMetadata_C, prepare_batch_infos

class BaseBlockManager:
    """Base class for block managers"""
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, max_running_reqs: int, device: str = "cuda"):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.max_running_reqs = max_running_reqs
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.decode_seq_lens = {}  # Track sequence lengths for each request
        self.query_start_loc = torch.arange(self.model_config.max_batch_size_attn + 1, dtype=torch.int32, device=self.device)
        
    def reset_state(self):
        pass
    
    def update_block_table(self, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata):
        pass
    
    def pack_flash_attn_metadata(self, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata, dummy_cache: bool = False) -> FlashAttentionMetadata:
        pass
    
    def release_seqs(self, seq_ids: List[int]):
        pass

class CPUBlockManager(BaseBlockManager):
    """CPU-based block manager for comparison - follows original implementation"""
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, max_running_reqs: int, device: str = "cuda"):
        super().__init__(model_config, cache_config, max_running_reqs, device)
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self._block_mgr = BlockManager_C(self.block_size, self.num_gpu_blocks, 0)

    def reset_state(self):
        self.release_seqs(list(self.decode_seq_lens.keys()))
        self.decode_seq_lens = {}
    
    def release_seqs(self, seq_ids: List[int]):
        req_ids = [seq_id for seq_id in seq_ids if seq_id in self.decode_seq_lens]
        self._block_mgr.batch_release(req_ids)
        for req_id in req_ids:
            self.decode_seq_lens.pop(req_id)
    
    @nvtx_range("CPUBlockManager.update_block_table")
    def update_block_table(self, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata):
        init_seq_ids = meta_py.seq_ids[:meta_py.num_prefill_seqs]
        decode_seq_ids = meta_py.seq_ids
        
        # If the first layer in this attention worker, update block table and decode_seq_lens
        if meta_py.layer_id == self.model_config.layer_ids[0]:
            # Allocate kv blocks for init seqs, update for all decoding seqs
            for i, seq_id in enumerate(init_seq_ids):
                self.decode_seq_lens[seq_id] = meta_py.init_prefill_lens[i]
            
            decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]
            
            # Update block table
            self._block_mgr.update_block_table(meta_c, decode_seq_lens)
            
            # Increment sequence lengths for all sequences
            for i, seq_id in enumerate(decode_seq_ids):
                decode_seq_lens[i] += 1
                self.decode_seq_lens[seq_id] += 1
        else:
            decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]
            
        meta_py.seq_lens = decode_seq_lens
        
    @nvtx_range("CPUBlockManager.pack_flash_attn_metadata")
    def pack_flash_attn_metadata(
        self, 
        meta_c: AttentionBatchMetadata_C, 
        meta_py: AttentionBatchMetadata, 
        dummy_cache: bool = False
    ) -> FlashAttentionMetadata:
        """Pack FlashAttention metadata using CPU approach - follows original implementation"""
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens
        
        # 1. prepare block table
        if dummy_cache:
            # dummy_cache is True when _warmup_attn
            block_table_1d = torch.zeros(
                (num_tokens + num_seqs * self.model_config.max_seq_len // self.block_size, ), 
                dtype=torch.int32, device=self.device)
        else:
            block_table_1d = self._block_mgr.prepare_block_table(meta_c, meta_py.seq_lens)

        slot_mapping_cuda = block_table_1d[-num_tokens:].to(torch.int64)
        block_table_cuda = block_table_1d[:-num_tokens].view(num_tokens, -1)

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        batch_infos_cuda = prepare_batch_infos(meta_c, meta_py.seq_lens)
        
        seq_lens_cuda = batch_infos_cuda[ : num_seqs]
        context_lens_cuda = batch_infos_cuda[num_seqs : num_seqs + num_seqs]
        seq_start_loc_cuda = batch_infos_cuda[num_seqs + num_seqs : ]
            
        query_start_loc = self.query_start_loc[ : num_tokens + 1]
        max_decode_seq_len = max(meta_py.seq_lens) if len(meta_py.seq_lens) > 0 else 0
        
        meta_py.seq_lens_tensor = seq_lens_cuda
        
        return FlashAttentionMetadata(
            0,
            0,
            num_tokens,
            slot_mapping_cuda,
            seq_lens=meta_py.seq_lens,
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
        
    def release_seqs(self, seq_ids: List[int]):
        req_indices = [self.req_to_indice.get(i) for i in seq_ids if i in self.decode_seq_lens]
        req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device=self.device)
        self.req_to_token_pool.free(req_indices)
        # get_logger().info(f"releasing seqs {seq_ids}")

        self.token_allocator.free_group_begin()
        for seq_id, req_indice in zip(seq_ids, req_indices):
            # NOTE: single read/write to python dict is thread-safe due to GIL, but iterating should be protected by a lock
            seq_len = self.decode_seq_lens[seq_id]
            kv_indices = self.req_to_token_pool.req_to_token[req_indice, : seq_len]
            self.token_allocator.free(kv_indices)
            self.decode_seq_lens.pop(seq_id)
        self.token_allocator.free_group_end()
    
    @nvtx_range("GPUBlockManager.update_block_table")
    def update_block_table(self, meta_c: AttentionBatchMetadata_C, meta_py: AttentionBatchMetadata):
        init_seq_ids = meta_py.seq_ids[:meta_py.num_prefill_seqs]
        running_seq_ids = meta_py.seq_ids[meta_py.num_prefill_seqs:]
        seq_ids = meta_py.seq_ids
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        
        if meta_py.layer_id == 0:  # First layer
            # Allocate prefill token slots for init seqs
            new_req_indices = self.req_to_token_pool.alloc(meta_py.num_prefill_seqs)
            for i, seq_id in enumerate(init_seq_ids):
                req_indice = new_req_indices[i]
                self.req_to_indice[seq_id] = req_indice
                prefill_kv_locs = self.token_allocator.alloc(meta_py.init_prefill_lens[i])
                self.req_to_token_pool.write((req_indice, slice(0, meta_py.init_prefill_lens[i])), prefill_kv_locs)
                self.decode_seq_lens[seq_id] = meta_py.init_prefill_lens[i]
                
            running_req_indices = [self.req_to_indice.get(seq_id) for seq_id in running_seq_ids]
            batch_req_indices = running_req_indices + new_req_indices
            batch_req_indices_tensor = torch.tensor(batch_req_indices, dtype=torch.int32, device=self.device)
            
            seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in seq_ids]
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
            
            for i, seq_id in enumerate(seq_ids):
                seq_lens[i] += 1
                self.decode_seq_lens[seq_id] += 1
                
            increment_locs = self.token_allocator.alloc(num_tokens)
            self.req_to_token_pool.write_loc(batch_req_indices_tensor, seq_lens_tensor, increment_locs.to(torch.int32))
            seq_lens_tensor = seq_lens_tensor + 1
            self.req_seq_lens[batch_req_indices_tensor] = seq_lens_tensor
        else:
            seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in seq_ids]
            batch_req_indices = [self.req_to_indice.get(seq_id) for seq_id in seq_ids]
            batch_req_indices_tensor = torch.tensor(batch_req_indices, dtype=torch.int32, device=self.device)
            seq_lens_tensor = self.req_seq_lens[batch_req_indices_tensor]
            
        meta_py.seq_lens = seq_lens
        meta_py.seq_lens_tensor = seq_lens_tensor
        meta_py.req_indices = batch_req_indices
        meta_py.req_indices_tensor = batch_req_indices_tensor
        
        return seq_lens
    
    @nvtx_range("GPUBlockManager.pack_flash_attn_metadata")
    def pack_flash_attn_metadata(
            self, 
            meta_c: AttentionBatchMetadata, 
            meta_py: AttentionBatchMetadata, 
            dummy_cache: bool = False
        ) -> FlashAttentionMetadata:
        """Pack FlashAttention metadata using GPU approach"""
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens

        seq_lens_cuda = meta_py.seq_lens_tensor
        context_lens_cuda = seq_lens_cuda - 1
        torch.cumsum(seq_lens_cuda, dim=0, out=self.seq_start_loc[1 : num_tokens + 1])
        seq_start_loc_cuda = self.seq_start_loc[ : num_tokens + 1]
        query_start_loc = self.query_start_loc[ : num_tokens + 1]
        seq_lens = meta_py.seq_lens
        max_decode_seq_len = max(seq_lens) if len(seq_lens) > 0 else 0
        
        if dummy_cache:
            block_table_cuda = torch.arange(
                num_seqs * max_decode_seq_len // self.cache_config.block_size, 
                dtype=torch.int32, device=self.device
            ).view(num_seqs, -1)
            slot_mapping_cuda = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        else:
            req_indices = meta_py.req_indices_tensor
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
    