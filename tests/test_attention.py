import torch
from disagmoe.models.attention import MoEAttention
from vllm.config import CacheConfig
from typing import Union
import random
from vllm.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadata
from torch.nn.utils.rnn import pad_sequence
def make_kv_cache(num_blocks: int,
                  num_heads: int,
                  head_size: int,
                  block_size: int,
                  device: Union[torch.device, str] = None,
                  default_val: float = 0.0) -> torch.Tensor:
    '''
    Create a fake KV cache.

    Arguments:

    * num_blocks: number of blocks in the KV cache
    * num_heads: number of attention heads
    * head_size: head dimension
    * block_size: number of offsets within a block
    * device: CPU or CUDA device
    * default_val: initialization value for KV cache elements

    Returns:

    * kv_cache: 2 x num_blocks x (block_size * num_heads * head_size)
    '''
    if device == None:
        device = torch.get_default_device()
        
    kv_cache = torch.rand(
        (2, num_blocks, block_size, num_heads, head_size)).to(device)
    if default_val is not None:
        kv_cache[:, :, :] = default_val
    return kv_cache



torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.bfloat16)

hidden_size = 1024
num_heads = 16
head_size = hidden_size // num_heads
num_kv_heads = 8
num_experts = 8
max_seq_len = 1024

block_size = 32

cache_conf = CacheConfig(block_size, 0.8, 2, "auto")
cache = make_kv_cache(2**10, num_kv_heads, head_size, block_size)

attn = MoEAttention(hidden_size, num_heads, num_kv_heads, num_experts, cache_config=cache_conf)

def make_seqlens(lens):
    seqlen = [0]
    for l in lens:
        seqlen.append(seqlen[-1] + l)
    return torch.tensor(seqlen, dtype=torch.int32, device=torch.get_default_device())

def make_naive_mapping(lens, mode):
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

def test_prefill():
    num_prefills = 8
    lens = [random.randint(32, b=127) for _ in range(num_prefills)]
    seqlens = torch.tensor(lens)
    num_prefill_tokens = sum(lens)
    seqlens = torch.tensor(lens, dtype=torch.int32, device=torch.get_default_device())
    seqlens_q = make_seqlens(lens)
    context_lens_tensor = [0] * num_prefills
    seqlens_kv = seqlens_q
    max_seqlen_q = max(lens)
    max_seqlen_kv = max_seqlen_q
    block_table, slot_mapping = make_naive_mapping(lens, "prefill")
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
    inputs = torch.randn((num_prefill_tokens, hidden_size))
    positions = torch.zeros_like(inputs, dtype=torch.long)
    attn.forward(positions, inputs, cache, meta)
    print(f">>> prefill test passed")
    

def test_decode():
    num_decode_tokens = 8
    num_prefills = 0
    num_prefill_tokens = 0
    lens = [random.randint(32, b=127) for _ in range(num_decode_tokens)]
    seqlens = torch.tensor(lens, dtype=torch.int32, device=torch.get_default_device())
    max_decode_seq_len = max(lens)
    block_table, slot_mapping = make_naive_mapping(lens, "decode")
    meta = FlashAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        slot_mapping=slot_mapping,
        seq_lens=lens,
        seq_lens_tensor=seqlens,
        max_query_len=None,
        max_prefill_seq_len=0,
        max_decode_seq_len=max_decode_seq_len,
        query_start_loc=None,
        seq_start_loc=None,
        context_lens_tensor=None,
        block_tables=block_table,
        use_cuda_graph=False,
    )
    inputs = torch.randn((num_decode_tokens, hidden_size))
    positions = torch.zeros_like(inputs, dtype=torch.long)
    attn.forward(positions, inputs, cache, meta)
    print(f">>> decode test passed")
    

def test_prefill_decode():
    num_decode_tokens = 8
    num_prefills = 8
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
    inputs = torch.randn((num_decode_tokens + num_prefill_tokens, hidden_size))
    positions = torch.zeros_like(inputs, dtype=torch.long)
    attn.forward(positions, inputs, cache, meta)
    print(f">>> chunked_prefill test passed")
    
test_prefill()

test_decode()

test_prefill_decode()