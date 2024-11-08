from disagmoe_c import AttentionBatchMetadata as AttentionBatchMetadata_C
from disagmoe.frontend.datatypes import AttentionBatchMetadata

shape = (1, 4096)
meta = AttentionBatchMetadata(
    0, 
    shape,
    "fp16",
    1,
    1,
    0,
    [0],
    [1],
    [1],
    []
)

attn_meta = AttentionBatchMetadata_C()
attn_meta.layer_id = meta.layer_id
attn_meta.shape = meta.shape
attn_meta.dtype = meta.dtype
attn_meta.num_prefill_seqs = meta.num_prefill_seqs
attn_meta.num_prefill_tokens = meta.num_prefill_tokens
attn_meta.num_decode_tokens = meta.num_decode_tokens
attn_meta.seq_ids = meta.seq_ids
attn_meta.prefill_seq_len = meta.prefill_seq_len
attn_meta.prefill_query_len = meta.prefill_query_len
attn_meta.expert_ids = meta.expert_ids
meta = attn_meta

print(meta)