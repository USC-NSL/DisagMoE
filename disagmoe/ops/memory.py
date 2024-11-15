import torch

import triton
import triton.language as tl

from disagmoe.utils.utils import nvtx_range
from typing import List, Tuple, Union
from disagmoe_c import permute_tokens_cuda as _permute_tokens_cuda
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
def get_mappings_from_exp_ids(exp_ids: Union[torch.Tensor, List[int]], num_experts: int) -> Tuple[List[int], List[int]]:
    if torch.is_tensor(exp_ids):
        exp_ids = exp_ids.view(-1).tolist()
    
    exp_cnt = [0] * num_experts
    exp_cumsum = [0] * num_experts
    
    for id in exp_ids:
        exp_cnt[id] += 1
        
    exp_cumsum[0] = exp_cnt[0]
    for i in range(1, num_experts):
        exp_cumsum[i] = exp_cumsum[i-1] + exp_cnt[i]
    
    mappings = [0] * len(exp_ids)
    
    for i, id in enumerate(exp_ids):
        exp_cumsum[id] -= 1
        mappings[i] = exp_cumsum[id]
        
    return mappings, exp_cnt

@nvtx_range("permute_tokens_triton")
def permute_tokens_triton(tokens: torch.Tensor, 
                   mappings: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    # permute tokens according to its expert id
    assert len(tokens.shape) == 2 # [num_tokens, hidden_size]
    num_tokens, hiddens_size = tokens.shape
    assert(tokens.is_contiguous())
    permuted_tokens = torch.empty((num_tokens, hiddens_size), device=tokens.device, dtype=tokens.dtype)
    
    if not torch.is_tensor(mappings):
        mappings = torch.tensor(mappings, dtype=torch.int32, device=tokens.device)
    
    grid = lambda META: (num_tokens, triton.cdiv(hiddens_size, META["BLOCK_SIZE"]))    
    _permute_tokens_kernel[grid](
        permuted_tokens, 
        tokens, 
        mappings.to(tokens.device),
        hiddens_size,
        BLOCK_SIZE=1024
    )
    return permuted_tokens

@nvtx_range("permute_tokens_cuda")
def permute_tokens_cuda(tokens: torch.Tensor, 
                   mappings: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    if not torch.is_tensor(mappings):
        mappings = torch.tensor(mappings, dtype=torch.int32, device=tokens.device)
    return _permute_tokens_cuda(tokens, mappings.to(tokens.device))