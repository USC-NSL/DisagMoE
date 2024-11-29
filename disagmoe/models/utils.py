from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from typing import Tuple
import torch.distributed as dist

import torch
import pickle
from typing import List, Any, Optional
import numpy as np

def pack_flash_attn_meta(buffer_meta: torch.Tensor,
                         buffer_locs: torch.Tensor,
                         layer_id: int, 
                         meta: FlashAttentionMetadata):
    num_seqs = meta.num_prefills + meta.num_decode_tokens
    buffer_meta[0] = layer_id
    buffer_meta[1] = meta.num_prefills
    buffer_meta[2] = meta.num_prefill_tokens
    buffer_meta[3] = meta.num_decode_tokens
    buffer_meta[4] = meta.block_tables.shape[1]  # max_blocks_per_seq
    assert meta.seq_lens_tensor.shape[0] == num_seqs, \
        f"{meta.seq_lens_tensor.shape[0]} != {num_seqs}"
    buffer_meta[5: 5 + num_seqs] = meta.seq_lens_tensor
    
    offset = 5 + num_seqs
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
    
    if meta.num_prefills > 0:
        buffer_locs[0][0: meta.num_prefills + 1] = meta.query_start_loc
    buffer_locs[1][0: num_seqs + 1] = meta.seq_start_loc

def unpack_flash_attn_meta(buffer_meta: torch.Tensor,
                           buffer_locs: torch.Tensor) -> Tuple[int, int, FlashAttentionMetadata]:
    layer_id = int(buffer_meta[0].item())
    num_prefills = int(buffer_meta[1].item())
    num_prefill_tokens = int(buffer_meta[2].item())
    num_decode_tokens = int(buffer_meta[3].item())
    max_blocks_per_seq = int(buffer_meta[4].item())
    num_seqs = num_prefills + num_decode_tokens
    
    seq_lens_tensor = buffer_meta[5: 5 + num_seqs]
    seq_lens = seq_lens_tensor.tolist()
    
    offset = 5 + num_seqs
    max_query_len = int(buffer_meta[offset + 0].item())
    max_prefill_seq_len = int(buffer_meta[offset + 1].item())
    max_decode_seq_len = int(buffer_meta[offset + 2].item())
    context_lens_tensor = buffer_meta[offset + 3: offset + 3 + num_seqs]
    
    offset += 3 + num_seqs
    
    if num_prefills > 0:
        query_start_loc = buffer_locs[0][0: num_prefills + 1]
    else:
        query_start_loc = None
    seq_start_loc = buffer_locs[1][0: num_seqs + 1]
    
    return layer_id, max_blocks_per_seq, FlashAttentionMetadata(
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
    
def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""

    if rank == 0:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            )
            tensor_size = torch.tensor([size], dtype=torch.long)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data