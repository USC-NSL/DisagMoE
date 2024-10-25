from disagmoe.frontend.engine import Engine, AttentionBatchMetadata, FlashAttentionMetadata
from disagmoe.utils.constants import *

from disagmoe_c import AttentionBatchMetadata as AttentionBatchMetadata_C

import torch

torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.bfloat16)

engine = Engine()

engine.set_device_id(0)
engine.setup_engine(True)
engine.init_core([0], [], [], [], {})
engine.scheduler = engine.a_scheduler

engine.start_profile()
 
meta: AttentionBatchMetadata = AttentionBatchMetadata_C()

def test_mixed_batch():
    meta.layer_id = 0
    meta.shape = [1, HIDDEN_SIZE]
    meta.dtype = "fp16"
    meta.num_prefill_seqs = 1
    meta.num_prefill_tokens = 1
    meta.num_decode_tokens = 0
    meta.seq_ids = [0]
    meta.prefill_seq_len = [1]
    meta.prefill_query_len = [1]
    meta.expert_ids = [0]

    tensor, new_meta = engine.process_batch_attn(
        meta,
        torch.Tensor(size=(1, HIDDEN_SIZE)).type(torch.bfloat16).cuda(),
        mocking=True
    )

    print(tensor)
    print(new_meta)

    meta.shape = [2, HIDDEN_SIZE]
    meta.num_decode_tokens = 1
    meta.seq_ids = [1, 0]

    tensor, new_meta = engine.process_batch_attn(
        meta, 
        torch.Tensor(size=(2, HIDDEN_SIZE)).type(torch.bfloat16).cuda(),
        mocking=True
    )

    print(tensor)
    print(new_meta)
    
def test_prefill():
    meta.layer_id = 0
    meta.shape = [2, HIDDEN_SIZE]
    meta.dtype = "fp16"
    meta.num_prefill_seqs = 2
    meta.num_prefill_tokens = 2
    meta.num_decode_tokens = 0
    meta.seq_ids = [0, 1]
    meta.prefill_seq_len = [1, 1]
    meta.prefill_query_len = [1, 1]
    meta.expert_ids = [0, 0]

    tensor, new_meta = engine.process_batch_attn(
        meta,
        torch.Tensor(size=meta.shape).type(torch.bfloat16).cuda(),
        mocking=True
    )

    print(tensor)
    print(new_meta)

def test_decode():
    meta.layer_id = 0
    meta.shape = [1, HIDDEN_SIZE]
    meta.dtype = "fp16"
    meta.num_prefill_seqs = 1
    meta.num_prefill_tokens = 1
    meta.num_decode_tokens = 0
    meta.seq_ids = [0]
    meta.prefill_seq_len = [1]
    meta.prefill_query_len = [1]
    meta.expert_ids = [0]
    
    _, _ = engine.process_batch_attn(
        meta,
        torch.Tensor(size=meta.shape).type(torch.bfloat16).cuda(),
        mocking=True
    )
    print("finish prefill")
    
    meta.num_prefill_seqs = 0
    meta.num_prefill_tokens = 0
    meta.num_decode_tokens = 1
    meta.prefill_seq_len = []
    meta.prefill_query_len = []
    
    for i in range(33):
        print("decoding token", i+1)
        _, _ = engine.process_batch_attn(
            meta,
            torch.Tensor(size=meta.shape).type(torch.bfloat16).cuda(),
            mocking=True
        )
    
    print("passed")

# test_prefill()
# test_mixed_batch()
test_decode()

engine.stop_profile()
