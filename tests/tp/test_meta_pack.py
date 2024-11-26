from disagmoe.models.utils import pack_flash_attn_meta, unpack_flash_attn_meta
from disagmoe.frontend.engine import FlashAttentionMetadata
from torch.nn.utils.rnn import pad_sequence

import torch
import random

block_size = 32

def make_seqlens(lens):
    seqlen = [0]
    for l in lens:
        seqlen.append(seqlen[-1] + l)
    return torch.tensor(seqlen, dtype=torch.int32, device=torch.get_default_device())

def make_mix_mapping(seq_lens, query_lens, num_prefill, num_decode):
    # tail query of seqs
    assert len(seq_lens) == num_prefill + num_decode
    block_table = []
    slots_table = []
    allocated_blocks = 0
    
    for i, l in enumerate(seq_lens):
        num_blocks = (l + block_size) // block_size
        start = allocated_blocks
        end = num_blocks + allocated_blocks
        block_list = list(range(start, end))
        
        block_table.append(torch.tensor(block_list, dtype=torch.int32))
        if i < num_prefill:
            start_slot = start * block_size
            end_slot = start_slot + l
            slot_list = list(range(end_slot - query_lens[i], end_slot))
            slots_table.extend(slot_list)
        else:
            end_slot = start * block_size + l - 1
            slots_table.append(end_slot)
        
    block_table = pad_sequence(block_table, batch_first=True, padding_value=0)
    slots_table = torch.tensor(slots_table, dtype=torch.long)
    return block_table, slots_table

def main():
    torch.set_default_device("cuda")
    num_decode_tokens = 3
    num_prefills = 17
    lens = [random.randint(64, b=127) for _ in range(num_decode_tokens + num_prefills)]
    seqlens = torch.tensor(lens, dtype=torch.int32, device=torch.get_default_device())
    query_lens = [random.randint(20, 50) for _ in range(num_prefills)]
    max_query_len = max(query_lens)
    num_prefill_tokens = sum(query_lens)
    max_prefill_seq_len = max(lens[:num_prefills])
    max_decode_seq_len = max(lens[num_prefills:])
    query_start_loc = make_seqlens(query_lens)
    seq_start_loc = make_seqlens(lens)
    block_table, slot_mapping = make_mix_mapping(lens, query_lens, num_prefills, num_decode_tokens)
    context_lens_tensor = [lens[i] - query_lens[i] for i in range(num_prefills)] + [lens[i + num_prefills] - 1 for i in range(num_decode_tokens)]
    meta = FlashAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        slot_mapping=slot_mapping,
        seq_lens=lens,
        seq_lens_tensor=seqlens,
        max_query_len=max_query_len,
        max_prefill_seq_len=max_prefill_seq_len,
        max_decode_seq_len=max_decode_seq_len,
        query_start_loc=query_start_loc,
        seq_start_loc=seq_start_loc,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_table,
        use_cuda_graph=False,
    )
    buffer_meta = torch.zeros([8192], dtype=torch.int32)
    buffer_loc = torch.zeros([2, 8192], dtype=torch.int32)
    
    pack_flash_attn_meta(buffer_meta, buffer_loc, 0, meta)
    layer_id, unpacked_meta = unpack_flash_attn_meta(buffer_meta, buffer_loc)
    
    assert layer_id == 0
    assert unpacked_meta.num_prefills == num_prefills
    assert unpacked_meta.num_prefill_tokens == num_prefill_tokens
    assert unpacked_meta.num_decode_tokens == num_decode_tokens
    assert torch.equal(unpacked_meta.seq_lens_tensor, seqlens)
    assert unpacked_meta.max_query_len == max_query_len
    assert unpacked_meta.max_prefill_seq_len == max_prefill_seq_len
    assert unpacked_meta.max_decode_seq_len == max_decode_seq_len
    assert torch.equal(unpacked_meta.context_lens_tensor, torch.tensor(context_lens_tensor, dtype=torch.int32))
    assert torch.equal(unpacked_meta.query_start_loc, query_start_loc)
    assert torch.equal(unpacked_meta.seq_start_loc, seq_start_loc)
    # assert torch.equal(unpacked_meta.block_tables, block_table)
    # assert torch.equal(unpacked_meta.slot_mapping, slot_mapping)
    print("test passed")
    
main()