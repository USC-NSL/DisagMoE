import torch
import os

from torch import Tensor

from typing import List, Dict, Any, Optional

from vllm.attention.backends.flashinfer import FlashInferMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.abstract import AttentionMetadata

from disagmoe.frontend.datatypes import AttentionBatchMetadata
from disagmoe.config import ModelConfig, CacheConfig

from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    
from disagmoe_c import prepare_batch_infos
        
def pack_flash_attn_metadata(
    block_mgr,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    meta_c: AttentionBatchMetadata,
    meta_py: AttentionBatchMetadata,
    decode_seq_lens: List[int],
    mocking: bool = False,
) -> AttentionMetadata:
    
    num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
    num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens

    # print(f"meta_py {meta_py}, decode_seq_lens {self.decode_seq_lens}")
    
    # 1. prepare block table
    if not mocking:
        block_table_1d = block_mgr.prepare_block_table(meta_c, decode_seq_lens)
    else:
        # mocking=True when _warmup_attn
        block_table_1d = torch.zeros(
            (num_tokens + num_seqs * model_config.max_seq_len // cache_config.block_size, ), 
            dtype=torch.int32, device="cuda")

    slot_mapping_cuda = block_table_1d[-num_tokens : ].to(torch.int64)
    block_table_cuda = block_table_1d[ : -num_tokens].view(num_tokens, -1)

    # 2. prepare seqlens and start_locs
    # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
    batch_infos_cuda = prepare_batch_infos(meta_c, decode_seq_lens)

    seq_lens_cuda, context_lens_cuda, seq_start_loc_cuda = \
        torch.split(batch_infos_cuda, [num_seqs, num_seqs, num_seqs + 1], dim=0)

    seq_lens = decode_seq_lens
        
    max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
    
    max_num_blocks = (max(seq_lens) - 1) // cache_config.block_size + 1
    assert mocking or model_config.enable_cuda_graph_attn or \
            max_num_blocks == block_table_cuda.shape[-1], f"block table wrong, {meta_py}, {block_table_cuda.shape}, {block_table_1d.shape}"
    
    return FlashAttentionMetadata(
        0,
        0,
        num_tokens,
        slot_mapping_cuda,
        seq_lens=seq_lens,
        seq_lens_tensor=seq_lens_cuda,
        max_query_len=0,
        max_prefill_seq_len=0,
        max_decode_seq_len=max_decode_seq_len,
        query_start_loc=None,
        seq_start_loc=seq_start_loc_cuda,
        context_lens_tensor=context_lens_cuda,
        block_tables=block_table_cuda,
        use_cuda_graph=model_config.enable_cuda_graph_attn,
    )

_flash_infer_prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper = None
_flash_infer_decode_wrapper: BatchDecodeWithPagedKVCacheWrapper = None
_flash_infer_workspace_buffer: torch.Tensor = None

def get_flash_infer_workspace_buffer():
    global _flash_infer_workspace_buffer
    if _flash_infer_workspace_buffer is None:
        _flash_infer_workspace_buffer = torch.empty(
            (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
    return _flash_infer_workspace_buffer

def _get_flash_infer_prefill_wrapper():
    global _flash_infer_prefill_wrapper
    if _flash_infer_prefill_wrapper is None:
        _flash_infer_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            get_flash_infer_workspace_buffer(),
            "NHD"
        )
    return _flash_infer_prefill_wrapper

def _get_flash_infer_decode_wrapper():
    global _flash_infer_decode_wrapper
    if _flash_infer_decode_wrapper is None:
        _flash_infer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            get_flash_infer_workspace_buffer(),
            "NHD"
        )
    return _flash_infer_decode_wrapper

def pack_flash_infer_metadata(
        block_mgr,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        meta_c: AttentionBatchMetadata,
        meta_py: AttentionBatchMetadata,
        decode_seq_lens: List[int],
        mocking: bool = False,
) -> AttentionMetadata:
    num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
    num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens

    # print(f"meta_py {meta_py}, decode_seq_lens {self.decode_seq_lens}")
    
    paged_kv_indptr_tensor = torch.empty((0,), dtype=torch.int, device="cpu")
    paged_kv_indices_tensor = torch.empty((0,), dtype=torch.int, device="cpu")
    paged_kv_last_page_len_tensor = torch.empty((0,), dtype=torch.int, device="cpu")
    
    # 1. prepare block table
    if not mocking:
        block_table_1d = block_mgr.prepare_block_table_with_paged_indices(
            meta_c, decode_seq_lens,
            paged_kv_indices_tensor,
            paged_kv_indptr_tensor,
            paged_kv_last_page_len_tensor,
        )
        block_table_bound_tensor = torch.zeros(paged_kv_indptr_tensor.numel() -
                                                1,
                                                device=paged_kv_indptr_tensor.device,
                                                dtype=torch.int)
    else:
        # mocking=True when _warmup_attn
        block_table_1d = torch.zeros(
            (num_tokens + num_seqs * model_config.max_seq_len // cache_config.block_size, ), 
            dtype=torch.int32, device="cuda")
        paged_kv_indptr_tensor = None
        paged_kv_indices_tensor = None
        paged_kv_last_page_len_tensor = None
        block_table_bound_tensor = None

    slot_mapping_cuda = block_table_1d[-num_tokens : ].to(torch.int64)
    block_table_cuda = block_table_1d[ : -num_tokens].view(num_tokens, -1)

    # 2. prepare seqlens and start_locs
    # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
    batch_infos_cuda = prepare_batch_infos(meta_c, decode_seq_lens)

    seq_lens_cuda, context_lens_cuda, seq_start_loc_cuda = \
        torch.split(batch_infos_cuda, [num_seqs, num_seqs, num_seqs + 1], dim=0)

    seq_lens = decode_seq_lens
            
    max_num_blocks = (max(seq_lens) - 1) // cache_config.block_size + 1
    assert mocking or model_config.enable_cuda_graph_attn or \
            max_num_blocks == block_table_cuda.shape[-1], f"block table wrong, {meta_py}, {block_table_cuda.shape}, {block_table_1d.shape}"
                
    return FlashInferMetadata(
        num_prefills=0,
        slot_mapping=slot_mapping_cuda,
        num_prefill_tokens=0,
        num_decode_tokens=meta_py.num_decode_tokens,
        max_prefill_seq_len=0,
        block_tables=block_table_cuda,
        paged_kv_indptr=paged_kv_indptr_tensor,
        paged_kv_indices=paged_kv_indices_tensor,
        paged_kv_last_page_len=paged_kv_last_page_len_tensor,
        block_table_bound=block_table_bound_tensor,
        seq_lens_tensor=seq_lens_cuda,
        num_qo_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.hidden_size // model_config.num_heads,
        page_size=cache_config.block_size,
        seq_start_loc=seq_start_loc_cuda,
        query_start_loc=None,
        device=torch.get_default_device(),
        data_type=cache_config.cache_dtype,
        q_data_type=model_config.dtype,
        use_cuda_graph=model_config.enable_cuda_graph_attn,
        is_profile_run=mocking,
        
        decode_wrapper=_get_flash_infer_decode_wrapper(),
        prefill_wrapper=_get_flash_infer_prefill_wrapper(),
    )


pack_attn_metadata = pack_flash_infer_metadata if os.environ.get("VLLM_ATTENTION_BACKEND", "FLASHINFER") == "FLASHINFER" \
                       else pack_flash_attn_metadata