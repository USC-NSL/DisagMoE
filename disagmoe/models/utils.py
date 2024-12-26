from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Optional
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
    

def make_seqlens(lens: List[int]):
    seqlen = [0]
    for l in lens:
        seqlen.append(seqlen[-1] + l)
    return torch.tensor(seqlen, dtype=torch.int32, device=torch.get_default_device())

def make_naive_mapping(block_size, lens, mode):
    block_table = []
    slots_table = []
    allocated_blocks = 4
    for l in lens:
        num_blocks = (l + block_size) // block_size
        start = allocated_blocks
        end = num_blocks + allocated_blocks
        block_list = list(range(start, end))
        allocated_blocks = end
        block_table.append(torch.tensor(block_list, dtype=torch.int32))
        if mode == "prefill":
            start_slot = start * block_size
            end_slot = start_slot + l
            slots_list = list(range(start_slot, end_slot))
            slots_table.extend(slots_list)
        elif mode == "decode":
            end_slot = start * block_size + l - 1
            slots_table.append(end_slot)
        else:
            assert False
            
    block_table = pad_sequence(block_table, batch_first=True, padding_value=0)
    slots_table = torch.tensor(slots_table, dtype=torch.long)
    return block_table, slots_table

def make_prefill_meta(num_prefills: int, block_size: int):
    lens = [1 for _ in range(num_prefills)]
    seqlens = torch.tensor(lens)
    num_prefill_tokens = sum(lens)
    seqlens = torch.tensor(lens, dtype=torch.int32, device=torch.get_default_device())
    seqlens_q = make_seqlens(lens)
    context_lens_tensor = [0] * num_prefills
    seqlens_kv = seqlens_q
    max_seqlen_q = max(lens)
    max_seqlen_kv = max_seqlen_q
    block_table, slot_mapping = make_naive_mapping(block_size, lens, "prefill")
    meta = FlashAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=0,
        slot_mapping=slot_mapping,
        seq_lens=lens,
        seq_lens_tensor=seqlens,
        max_query_len=max_seqlen_q,
        max_prefill_seq_len=max_seqlen_q,
        max_decode_seq_len=0,
        query_start_loc=seqlens_q,
        seq_start_loc=seqlens_kv,
        context_lens_tensor=context_lens_tensor,
        block_tables=torch.tensor([]),
        use_cuda_graph=False,
    )
    return meta

def make_dummy_meta(num_prefill_tokens: int, num_decode_tokens: int):
    bs = num_prefill_tokens + num_decode_tokens
    meta = AttentionBatchMetadata(
        0,
        [num_decode_tokens + num_prefill_tokens, 1],
        "bf16",
        num_prefill_tokens,
        num_prefill_tokens,
        num_decode_tokens,
        [0] * bs,
        [0] * bs,
        [0] * bs,
        [0] * bs,
        [0] * bs,
    )
    return meta

@dataclass
class CudaGraphContext:
    graph: torch.cuda.CUDAGraph
    # static_input: torch.Tensor
    # static_output: torch.Tensor
    
    # # attn buffers
    # static_expert_ids: Optional[torch.Tensor] = None
    # static_positions: Optional[torch.Tensor] = None
    
    # # expert buffers
    # static_expert_mappings: Optional[torch.Tensor] = None

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
