import torch
import triton
import triton.language as tl
import numpy as np
from typing import List
from dataclasses import dataclass
from disagmoe.utils.tensor_utils import get_cuda_aligned_tensor
from disagmoe.utils.utils import nvtx_range
from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.frontend.datatypes import AttentionScheduleBatch
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from disagmoe.block_manager.mem_pool import ReqToTokenPool, TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator

from disagmoe_c import BlockManager as BlockManager_C, BatchMetadata as BatchMetadata_C, rebind_batch_info_tensor
from disagmoe.frontend.engine_utils import get_global_engine_config

from disagmoe.utils.gdr_context import GdrContext, use_gdrcopy_optimization

@dataclass
class BatchTensorBuffer:
    
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    seq_start_loc: torch.Tensor
    query_start_loc: torch.Tensor
    
    block_table_view: torch.Tensor
    slot_mapping_view: torch.Tensor
    seq_lens_view: torch.Tensor
    context_lens_view: torch.Tensor
    seq_start_loc_view: torch.Tensor
    query_start_loc_view: torch.Tensor
    
    block_table_gdr: GdrContext
    slot_mapping_gdr: GdrContext
    seq_lens_gdr: GdrContext
    context_lens_gdr: GdrContext
    seq_start_loc_gdr: GdrContext
    
    def __init__(self, max_batch_size: int, max_num_pages: int, device: str = "cuda"):
        self.block_table = get_cuda_aligned_tensor(max_batch_size * max_num_pages, torch.int32, device=device)
        self.slot_mapping = get_cuda_aligned_tensor(max_batch_size, torch.int64, device=device)
        self.seq_lens = get_cuda_aligned_tensor(max_batch_size, torch.int32, device=device)
        self.context_lens = get_cuda_aligned_tensor(max_batch_size, torch.int32, device=device)
        self.seq_start_loc = get_cuda_aligned_tensor(max_batch_size + 1, torch.int32, device=device)
        self.query_start_loc = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        
        self.block_table_gdr = GdrContext(self.block_table)
        self.slot_mapping_gdr = GdrContext(self.slot_mapping)
        self.seq_lens_gdr = GdrContext(self.seq_lens)
        self.context_lens_gdr = GdrContext(self.context_lens)
        self.seq_start_loc_gdr = GdrContext(self.seq_start_loc)
        
        self.block_table_view = torch.empty(0, dtype=torch.int32, device=device)
        self.slot_mapping_view = torch.empty(0, dtype=torch.int64, device=device)
        self.seq_lens_view = torch.empty(0, dtype=torch.int32, device=device)
        self.context_lens_view = torch.empty(0, dtype=torch.int32, device=device)
        self.seq_start_loc_view = torch.empty(0, dtype=torch.int32, device=device)
        self.query_start_loc_view = torch.empty(0, dtype=torch.int32, device=device)
        
    def create_view(self, num_tokens: int, num_pages: int):
        rebind_batch_info_tensor(
            num_tokens,
            num_pages,
            self.block_table_view,
            self.slot_mapping_view,
            self.seq_lens_view,
            self.context_lens_view,
            self.seq_start_loc_view,
            self.query_start_loc_view,
            self.block_table,
            self.slot_mapping,
            self.seq_lens,
            self.context_lens,
            self.seq_start_loc,
            self.query_start_loc
        )
        
class ReqManager:
    
    def __init__(self, max_running_reqs: int, use_list: bool = False):
        self.max_running_reqs = max_running_reqs
        self.use_list = use_list
        self.decode_seq_lens = {}
        self.decode_seq_lens_list = [0] * max_running_reqs
        
    def update_decode_seq_lens_dict(self, req_id: int, seq_len: int):
        self.decode_seq_lens[req_id] = seq_len
        
    def get_decode_seq_lens_dict(self, req_id: int) -> int:
        return self.decode_seq_lens[req_id]
    
    def get_active_req_ids_dict(self) -> List[int]:
        return list(self.decode_seq_lens.keys())
    
    def release_reqs_dict(self, req_ids: List[int]):
        for req_id in req_ids:
            self.decode_seq_lens.pop(req_id)
    
    def update_decode_seq_lens_list(self, req_id: int, seq_len: int):
        self.decode_seq_lens_list[req_id] = seq_len

    def get_decode_seq_lens_list(self, req_id: int) -> int:
        return self.decode_seq_lens_list[req_id]
    
    def get_active_req_ids_list(self) -> List[int]:
        req_ids = [i for i in range(self.max_running_reqs) if self.decode_seq_lens_list[i] > 0]
        return req_ids
    
    def release_reqs_list(self, req_ids: List[int]):
        for req_id in req_ids:
            self.decode_seq_lens_list[req_id] = 0
            
    def get_decode_seq_lens(self, req_id: int) -> int:
        if self.use_list:
            return self.get_decode_seq_lens_list(req_id)
        else:
            return self.get_decode_seq_lens_dict(req_id)
    
    def update_decode_seq_lens(self, req_id: int, seq_len: int):
        if self.use_list:
            self.update_decode_seq_lens_list(req_id, seq_len)
        else:
            self.update_decode_seq_lens_dict(req_id, seq_len)
            
    def get_active_req_ids(self) -> List[int]:
        if self.use_list:
            return self.get_active_req_ids_list()
        else:
            return self.get_active_req_ids_dict()
            
    def release_reqs(self, req_ids: List[int]):
        if self.use_list:
            self.release_reqs_list(req_ids)
        else:
            self.release_reqs_dict(req_ids)
            
    def reset(self):
        self.decode_seq_lens = {}
        self.decode_seq_lens_list = [0] * self.max_running_reqs

class BaseBlockManager:
    """Base class for block managers"""
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, max_running_reqs: int, device: str = "cuda"):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.max_running_reqs = max_running_reqs
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.decode_seq_lens = {}  # Track sequence lengths for each request
        self.query_start_loc_cuda_buffer = torch.arange(get_global_engine_config().max_batch_size_attn + 1, dtype=torch.int32, device=self.device)
        
    def reset_state(self):
        pass
    
    def update_block_table(self, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch):
        pass
    
    def pack_flash_attn_metadata(self, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch, dummy_cache: bool = False) -> FlashAttentionMetadata:
        pass
    
    def release_seqs(self, req_ids: List[int]):
        pass

class CPUBlockManager(BaseBlockManager):
    """CPU-based block manager for comparison - follows original implementation"""
    def __init__(
        self, 
        model_config: ModelConfig, 
        cache_config: CacheConfig, 
        max_running_reqs: int, 
        device: str = "cuda",
        use_rebind: bool = True,
    ):
        super().__init__(model_config, cache_config, max_running_reqs, device)
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self._block_mgr = BlockManager_C(self.block_size, self.num_gpu_blocks, 0)
        
        self.use_gdr_copy = use_gdrcopy_optimization
        self.use_rebind = use_rebind
        
        self.req_manager = ReqManager(max_running_reqs)
        
        max_pages_per_req = self.model_config.max_seq_len // self.cache_config.block_size
        
        self.batch_tensor_buffer = BatchTensorBuffer(get_global_engine_config().max_batch_size_attn, max_pages_per_req)
        self.batch_tensor_buffer_alt = BatchTensorBuffer(get_global_engine_config().max_batch_size_attn, max_pages_per_req)
    
    def reset_state(self):
        self.release_seqs(list(self.req_manager.get_active_req_ids()))
        self.req_manager.reset()
    
    def release_seqs(self, req_ids: List[int]):
        self._block_mgr.batch_release(req_ids)
        self.req_manager.release_reqs(req_ids)
    
    @nvtx_range("CPUBlockManager.update_block_table")
    def update_block_table(self, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch):
        init_req_ids = batch.req_ids[:batch.num_prefill_seqs]
        decode_req_ids = batch.req_ids
        
        # If the first layer in this attention worker, update block table and decode_seq_lens
        if batch.layer_id == self.model_config.layer_ids[0]:
            # Allocate kv blocks for init seqs, update for all decoding seqs
            for i, req_id in enumerate(init_req_ids):
                self.req_manager.update_decode_seq_lens(req_id, batch.init_prefill_lens[i])
            
            decode_seq_lens = [self.req_manager.get_decode_seq_lens(req_id) for req_id in decode_req_ids]
            
            # Update block table
            self._block_mgr.update_block_table(meta_c, decode_seq_lens)
            
            # Increment sequence lengths for all sequences
            for i, req_id in enumerate(decode_req_ids):
                decode_seq_lens[i] += 1
                self.req_manager.update_decode_seq_lens(req_id, decode_seq_lens[i])
        else:
            decode_seq_lens = [self.req_manager.get_decode_seq_lens(req_id) for req_id in decode_req_ids]
            
        batch.seq_lens = decode_seq_lens
        
    def swap_batch_tensor_buffer(self):
        self.batch_tensor_buffer, self.batch_tensor_buffer_alt = self.batch_tensor_buffer_alt, self.batch_tensor_buffer
        
    @nvtx_range("CPUBlockManager.pack_flash_attn_metadata")
    def pack_flash_attn_metadata(self, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch, dummy_cache: bool = False) -> FlashAttentionMetadata:
        self.swap_batch_tensor_buffer()
        if self.use_rebind and self.use_gdr_copy and not dummy_cache:
            return self.pack_flash_attn_metadata_opt(meta_c, batch)
        else:
            return self.pack_flash_attn_metadata_naive(meta_c, batch, dummy_cache)
    
    @nvtx_range("CPUBlockManager.pack_flash_attn_metadata_opt")
    def pack_flash_attn_metadata_opt(
        self, 
        meta_c: BatchMetadata_C, 
        batch: AttentionScheduleBatch
    ) -> FlashAttentionMetadata:
        """Pack FlashAttention metadata using CPU approach - follows original implementation"""
        num_tokens = batch.num_decode_tokens + batch.num_prefill_tokens
        
        num_pages_per_token = self._block_mgr.prepare_block_table_gdr(
            meta_c, batch.seq_lens,
            self.batch_tensor_buffer.block_table_gdr.gdr_context, 
            self.batch_tensor_buffer.slot_mapping_gdr.gdr_context
        )
        self._block_mgr.prepare_seq_info_gdr(
            meta_c, batch.seq_lens, 
            self.batch_tensor_buffer.seq_lens_gdr.gdr_context, 
            self.batch_tensor_buffer.context_lens_gdr.gdr_context, 
            self.batch_tensor_buffer.seq_start_loc_gdr.gdr_context
        )
        self.batch_tensor_buffer.create_view(num_tokens, num_pages_per_token)

        max_decode_seq_len = max(batch.seq_lens) if len(batch.seq_lens) > 0 else 0
        batch.seq_lens_tensor = self.batch_tensor_buffer.seq_lens_view
        
        return FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_tokens,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            max_decode_query_len=1,
            seq_lens=batch.seq_lens,
            block_tables=self.batch_tensor_buffer.block_table_view,
            slot_mapping=self.batch_tensor_buffer.slot_mapping_view,
            seq_lens_tensor=self.batch_tensor_buffer.seq_lens_view,
            context_lens_tensor=self.batch_tensor_buffer.context_lens_view,
            seq_start_loc=self.batch_tensor_buffer.seq_start_loc_view,
            query_start_loc=self.batch_tensor_buffer.query_start_loc_view,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
        )
        
    @nvtx_range("CPUBlockManager.pack_flash_attn_metadata_naive")
    def pack_flash_attn_metadata_naive(
        self, 
        meta_c: BatchMetadata_C, 
        batch: AttentionScheduleBatch, 
        dummy_cache: bool = False
    ) -> FlashAttentionMetadata:
        """Pack FlashAttention metadata using CPU approach - follows original implementation"""
        num_tokens = batch.num_decode_tokens + batch.num_prefill_tokens
        num_seqs = batch.num_prefill_seqs + batch.num_decode_tokens
        
        # 1. prepare block table
        if dummy_cache:
            # dummy_cache is True when _warmup_attn
            max_num_blocks = (max(batch.seq_lens) - 1) // self.block_size + 1
            block_table_cuda = torch.zeros(num_tokens * max_num_blocks, dtype=torch.int32, device=self.device).view(num_tokens, -1)
            slot_mapping_cuda = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        else:
            if self.use_gdr_copy:
                num_pages_per_token = self._block_mgr.prepare_block_table_gdr(
                    meta_c, batch.seq_lens, 
                    self.batch_tensor_buffer.block_table_gdr.gdr_context, 
                    self.batch_tensor_buffer.slot_mapping_gdr.gdr_context
                )
                block_table_cuda = self.batch_tensor_buffer.block_table[ : num_pages_per_token * num_tokens].view(num_tokens, -1)
                slot_mapping_cuda = self.batch_tensor_buffer.slot_mapping[ : num_tokens]
            else:
                block_table_1d = self._block_mgr.prepare_block_table(meta_c, batch.seq_lens)
                slot_mapping_cuda = block_table_1d[-num_tokens:].to(torch.int64)
                block_table_cuda = block_table_1d[:-num_tokens].view(num_tokens, -1)

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        if self.use_gdr_copy:
            self._block_mgr.prepare_seq_info_gdr(
                meta_c, batch.seq_lens, 
                self.batch_tensor_buffer.seq_lens_gdr.gdr_context, 
                self.batch_tensor_buffer.context_lens_gdr.gdr_context, 
                self.batch_tensor_buffer.seq_start_loc_gdr.gdr_context
            )
            seq_lens_cuda = self.batch_tensor_buffer.seq_lens[:num_seqs]
            context_lens_cuda = self.batch_tensor_buffer.context_lens[:num_seqs]
            seq_start_loc_cuda = self.batch_tensor_buffer.seq_start_loc[:num_seqs + 1]
        else:
            batch_infos_cuda = self._block_mgr.prepare_seq_info(meta_c, batch.seq_lens)
            seq_lens_cuda = batch_infos_cuda[ : num_seqs]
            context_lens_cuda = batch_infos_cuda[num_seqs : num_seqs + num_seqs]
            seq_start_loc_cuda = batch_infos_cuda[num_seqs + num_seqs : ]
                
        query_start_loc = self.batch_tensor_buffer.query_start_loc[ : num_tokens + 1]
        max_decode_seq_len = max(batch.seq_lens) if len(batch.seq_lens) > 0 else 0
        
        batch.seq_lens_tensor = seq_lens_cuda
        
        return FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_tokens,
            slot_mapping=slot_mapping_cuda,
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
        
        self.seq_start_loc = torch.zeros(get_global_engine_config().max_batch_size_attn + 1, dtype=torch.int32, device=self.device)
    
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
    def update_block_table(self, meta_c: BatchMetadata_C, batch: AttentionScheduleBatch):
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
            batch: AttentionScheduleBatch, 
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
    