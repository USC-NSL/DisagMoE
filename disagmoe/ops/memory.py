import torch

import triton
import triton.language as tl

from disagmoe.utils.utils import nvtx_range

@triton.jit
def _permute_tokens_kernel(
    out_ptr, # buffer for permuted tokens 
    in_ptr, # input tokens
    mapping,
    hidden_size, # int
    BLOCK_SIZE: tl.constexpr # division of hidden_size, should be tuned (default 128)
):
    token_id = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)

    target_pos = tl.load(mapping + token_id)

    src_start = token_id * hidden_size + block_id * BLOCK_SIZE
    src_offsets = src_start + tl.arange(0, BLOCK_SIZE)
    src_data = tl.load(in_ptr + src_offsets)
    
    target_start = target_pos * hidden_size + block_id * BLOCK_SIZE
    target_offsets = target_start + tl.arange(0, BLOCK_SIZE)
    
    tl.store(out_ptr + target_offsets, src_data)

@nvtx_range("get_mappings_from_exp_ids")
def get_mappings_from_exp_ids(exp_ids: torch.Tensor, num_experts: int):
    assert len(exp_ids.shape) == 1 # [num_tokens]
    
    exp_ids = exp_ids.to("cpu")
    exp_cnt = torch.bincount(exp_ids, minlength=num_experts)
    exp_cumsum = torch.cumsum(exp_cnt, dim=0)
    
    mappings = torch.empty(exp_ids.shape[0], device="cpu", dtype=torch.int32)  
    
    for i, id in enumerate(exp_ids):
        exp_cumsum[id] -= 1
        mappings[i] = exp_cumsum[id]
        
    return mappings, exp_cnt

@nvtx_range("permute_tokens")
def permute_tokens(tokens: torch.Tensor, 
                   mappings: torch.Tensor) -> torch.Tensor:
    # permute tokens according to its expert id
    assert len(tokens.shape) == 2 # [num_tokens, hidden_size]
    num_tokens, hiddens_size = tokens.shape
    assert(tokens.is_contiguous())
    permuted_tokens = torch.empty((num_tokens, hiddens_size), device=tokens.device, dtype=tokens.dtype)
    
    # print(f"token mapping by expert id: {mappings}")
    
    grid = lambda META: (num_tokens, triton.cdiv(hiddens_size, META["BLOCK_SIZE"]))    
    _permute_tokens_kernel[grid](
        permuted_tokens, 
        tokens, 
        mappings.cuda(),
        hiddens_size,
        BLOCK_SIZE=128
    )
    return permuted_tokens
