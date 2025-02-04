from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from disagmoe.utils.constants import MAX_SEQ_LEN
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from disagmoe.models.utils import make_dummy_meta
import torch
from typing import List
import time
from disagmoe_c import *

def pack_attn_metadata(
            meta_c: AttentionBatchMetadata,
            meta_py: AttentionBatchMetadata,
            decode_seq_lens: List[int],
        ) -> FlashAttentionMetadata:
        
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens

        block_size = 32

        block_table_1d = torch.zeros(
            (num_tokens + num_seqs * MAX_SEQ_LEN // block_size, ), 
            dtype=torch.int32, device="cuda")

        slot_mapping_cuda = block_table_1d[-num_tokens : ].to(torch.int64)
        block_table_cuda = block_table_1d[ : -num_tokens].view(num_tokens, -1)

        seq_lens = None
        seq_lens_cuda = None
        context_lens_tensor = None
        seq_start_loc = None
        max_decode_seq_len = 0

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        batch_infos_cuda = prepare_batch_infos(meta_c, decode_seq_lens)
        seq_lens_cuda = batch_infos_cuda[ : num_seqs]
        context_lens_tensor = batch_infos_cuda[num_seqs : num_seqs + num_seqs]
        seq_start_loc = batch_infos_cuda[num_seqs + num_seqs : ]
            
        # max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
        
        return FlashAttentionMetadata(
            0,
            0,
            meta_py.num_prefill_tokens + meta_py.num_decode_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=[],
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_table_cuda,
            use_cuda_graph=False,
        )

if __name__ == '__main__':
    batch_size = 256
    meta_py = make_dummy_meta(0, batch_size)

    def run():
        meta = pack_attn_metadata(meta_py.to_c(), meta_py, [256] * batch_size)

    for _ in range(2):
        run()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(5):
        run()

    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken: {(end - start) / 5 * 1000000:.1f} us")

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
