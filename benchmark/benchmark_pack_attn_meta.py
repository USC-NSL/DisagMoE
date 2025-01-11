from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from disagmoe.utils.constants import MAX_SEQ_LEN
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from disagmoe.models.utils import make_dummy_meta
from disagmoe.utils.utils import make_seqlens_list
import torch
from typing import List

def pack_attn_metadata(
            meta_c: AttentionBatchMetadata,
            meta_py: AttentionBatchMetadata,
            decode_seq_lens: List[int],
        ) -> FlashAttentionMetadata:
        
    num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
    num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens

    block_size = 16

    block_table_1d = torch.zeros(
        (num_tokens + num_seqs * MAX_SEQ_LEN // block_size, ), 
        dtype=torch.int32, device="cuda")
    
    slot_mapping_cuda = block_table_1d[-num_tokens : ].to(torch.int64)
    block_table_cuda = block_table_1d[ : -num_tokens].view(num_tokens, -1)

    # 2. prepare seqlens and start_locs
    # pack (seq_lens, context_lens, query_start_loc, seq_start_loc) in the same tensor
    batch_infos = [0] * (num_seqs + num_seqs + (meta_py.num_prefill_seqs + 1) + (num_seqs + 1)) 
    
    # make seq_lens
    batch_infos[ : meta_py.num_prefill_seqs] = meta_py.prefill_seq_len
    batch_infos[meta_py.num_prefill_seqs : num_seqs] = decode_seq_lens
    seq_lens = batch_infos[ : num_seqs]
    
    # make context_lens
    for i in range(meta_py.num_prefill_seqs):
        batch_infos[num_seqs + i] = meta_py.prefill_seq_len[i] - meta_py.prefill_query_len[i]
    for i in range(meta_py.num_decode_tokens):
        batch_infos[num_seqs + meta_py.num_prefill_seqs + i] = decode_seq_lens[i] - 1
        
    # make query_start_loc
    make_seqlens_list(meta_py.prefill_query_len, dst=batch_infos[num_seqs + num_seqs : num_seqs + num_seqs + meta_py.num_prefill_seqs + 1])

    # make seq_start_loc
    make_seqlens_list(seq_lens, dst=batch_infos[num_seqs + num_seqs + meta_py.num_prefill_seqs + 1 : ])

    batch_infos_cuda = torch.tensor(batch_infos, dtype=torch.int32, device="cuda")
    
    seq_lens_cuda = batch_infos_cuda[ : num_seqs]
    context_lens_tensor = batch_infos_cuda[num_seqs : num_seqs + num_seqs]
    query_start_loc = batch_infos_cuda[num_seqs + num_seqs : num_seqs + num_seqs + meta_py.num_prefill_seqs + 1]
    seq_start_loc = batch_infos_cuda[num_seqs + num_seqs + meta_py.num_prefill_seqs + 1 : ]

    max_query_len = max(meta_py.prefill_query_len) if len(meta_py.prefill_query_len) > 0 else 0
    max_prefill_seq_len = max(meta_py.prefill_seq_len) if len(meta_py.prefill_seq_len) > 0 else 0
    max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
    
    return FlashAttentionMetadata(
        meta_py.num_prefill_seqs,
        meta_py.num_prefill_tokens,
        meta_py.num_decode_tokens,
        slot_mapping_cuda,
        seq_lens=seq_lens,
        seq_lens_tensor=seq_lens_cuda,
        max_query_len=max_query_len,
        max_prefill_seq_len=max_prefill_seq_len,
        max_decode_seq_len=max_decode_seq_len,
        query_start_loc=query_start_loc,
        seq_start_loc=seq_start_loc,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_table_cuda,
        use_cuda_graph=True,
    )

if __name__ == '__main__':
    batch_size = 256
    meta_py = make_dummy_meta(batch_size, 0)

    def run():
        meta = pack_attn_metadata(meta_py.to_c(), meta_py, [])

    for _ in range(2):
        run()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./pack_attn_meta'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(5):
            run()
            prof.step()

