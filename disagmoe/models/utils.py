from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence

import torch

def pack_flash_attn_meta(buffer_meta: torch.Tensor,
                         buffer_locs: torch.Tensor,
                         layer_id: int, 
                         meta: FlashAttentionMetadata):
    num_seqs = meta.num_prefills + meta.num_decode_tokens
    buffer_meta[0] = layer_id
    buffer_meta[1] = meta.num_prefills
    buffer_meta[2] = meta.num_prefill_tokens
    buffer_meta[3] = meta.num_decode_tokens
    assert meta.seq_lens_tensor.shape[0] == num_seqs, \
        f"{meta.seq_lens_tensor.shape[0]} != {num_seqs}"
    buffer_meta[4: 4 + num_seqs] = meta.seq_lens_tensor
    
    offset = 4 + num_seqs
    buffer_meta[offset + 0] = meta.max_query_len
    buffer_meta[offset + 1] = meta.max_prefill_seq_len
    buffer_meta[offset + 2] = meta.max_decode_seq_len
    assert len(meta.context_lens_tensor) == num_seqs
    if isinstance(meta.context_lens_tensor, list):
        meta.context_lens_tensor = torch.tensor(meta.context_lens_tensor, dtype=torch.int32)
    buffer_meta[offset + 3: offset + 3 + num_seqs] = meta.context_lens_tensor
    offset += 3 + num_seqs
    
    # slot_mapping: shape (num_prefill_tokens + num_decode_tokens, )
    # blocktable: shape (num_seqs, max_blocks_per_seq)
    
    buffer_locs[0][0: meta.num_prefills + 1] = meta.query_start_loc
    buffer_locs[1][0: num_seqs + 1] = meta.seq_start_loc

def unpack_flash_attn_meta(buffer_meta: torch.Tensor,
                           buffer_locs: torch.Tensor) -> Tuple[int, FlashAttentionMetadata]:
    layer_id = int(buffer_meta[0].item())
    num_prefills = int(buffer_meta[1].item())
    num_prefill_tokens = int(buffer_meta[2].item())
    num_decode_tokens = int(buffer_meta[3].item())
    num_seqs = num_prefills + num_decode_tokens
    
    seq_lens_tensor = buffer_meta[4: 4 + num_seqs]
    seq_lens = seq_lens_tensor.tolist()
    
    offset = 4 + num_seqs
    max_query_len = int(buffer_meta[offset + 0].item())
    max_prefill_seq_len = int(buffer_meta[offset + 1].item())
    max_decode_seq_len = int(buffer_meta[offset + 2].item())
    context_lens_tensor = buffer_meta[offset + 3: offset + 3 + num_seqs]
    
    offset += 3 + num_seqs
    
    query_start_loc = buffer_locs[0][0: num_prefills + 1]    # TODO
    seq_start_loc = buffer_locs[1][0: num_seqs + 1]      # TODO
    
    return layer_id, FlashAttentionMetadata(
        num_prefills,
        num_prefill_tokens,
        num_decode_tokens,
        None, # slot_mapping
        seq_lens,
        seq_lens_tensor,
        max_query_len,
        max_prefill_seq_len,
        max_decode_seq_len,
        query_start_loc,
        seq_start_loc,
        context_lens_tensor,
        None, # block_table
        use_cuda_graph=False
    )