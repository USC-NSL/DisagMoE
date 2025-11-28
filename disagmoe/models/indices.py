import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
            mask=(start_m + off_expert) < cur_expert_token_num
        )

def get_m_indices(batch_sizes: torch.Tensor) -> torch.Tensor:
    """
    Generates m_indices by repeating expert IDs according to batch_sizes.
    """
    num_experts = batch_sizes.shape[0]
    total_tokens = batch_sizes.sum().item()
    
    # Ensure inputs are on the correct device and type
    if batch_sizes.dtype != torch.int32:
        batch_sizes = batch_sizes.to(torch.int32)
    
    device = batch_sizes.device
    m_indices = torch.empty(total_tokens, dtype=torch.int32, device=device)
    
    # Buffer for start locations (prefix sum)
    expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=device)
    
    # Triton kernel configuration
    BLOCK_EXPERT_NUM = triton.next_power_of_2(num_experts)
    BLOCK_E = 128
    
    grid = (num_experts,)
    
    _fwd_kernel_ep_scatter_1[grid](
        batch_sizes,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=BLOCK_EXPERT_NUM
    )
    
    return m_indices

