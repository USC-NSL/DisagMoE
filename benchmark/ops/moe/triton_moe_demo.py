import math
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the gated activation used in MoE MLPs: SiLU(gate) * up.

    Expects the last dimension of `x` to be interleaved as [gate, up, gate, up, ...].
    Returns the elementwise product of SiLU applied to the gate half and the up half.
    Shape is preserved.
    """
    gate = x[..., 0::2]
    up = x[..., 1::2]
    return F.silu(gate) * up


def _topk_router(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Softmax router logits and select top-k experts per token.

    Args:
        hidden_states: Unused here; present for API symmetry with typical routers.
        router_logits: Tensor of shape (M, E) with scores for E experts per token.
        top_k: Number of experts to route each token to.
        renormalize: If True, renormalize selected weights to sum to 1 across top-k.

    Returns:
        topk_weights: (M, top_k) probabilities for the chosen experts.
        topk_ids: (M, top_k) int32 indices of the chosen experts.
    """
    weights = F.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(weights, k=top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.to(torch.int32)


def _moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align token-expert assignments into BLOCK_SIZE_M-sized groups for the kernel.

    Given `topk_ids` of shape (M, top_k), flattens the M*top_k pairs, groups
    them by expert id, and pads each expert's group to a multiple of `block_size`
    using a sentinel index (M*top_k). Produces:

    - sorted_token_ids: 1D int32 indices into the flattened (M*top_k) array, padded
      with the sentinel. This defines the processing order of tokens.
    - expert_ids: 1D int32 list, one entry per block, indicating which expert the
      subsequent BLOCK_SIZE_M tokens correspond to (or the padding bucket).
    - num_tokens_post_padded: scalar int32 of the total rows in `sorted_token_ids`.

    This is a pure-Torch reference matching sglang-style alignment semantics.
    """
    # Pure Torch implementation matching sglang semantics
    # Flatten expert choices per token into length M*topk list
    m, topk = topk_ids.shape
    flat = topk_ids.view(-1)
    valid_mask = (flat >= 0) & (flat < num_experts)
    flat_valid = torch.where(valid_mask, flat, torch.full_like(flat, num_experts))

    # Collect indices per expert
    per_expert = [[] for _ in range(num_experts + 1)]  # +1 padding bucket
    for idx in range(flat_valid.numel()):
        e = int(flat_valid[idx])
        per_expert[e].append(idx)

    # Pad each expert's list to multiple of block_size with sentinel M*topk
    sentinel = m * topk
    sorted_ids_list = []
    expert_ids_list = []
    for e in range(num_experts + 1):
        ids = per_expert[e]
        if len(ids) == 0:
            continue
        # number of blocks for this expert
        num = len(ids)
        padded = ((num + block_size - 1) // block_size) * block_size
        ids = ids + [sentinel] * (padded - num)
        sorted_ids_list.extend(ids)
        expert_ids_list.extend([e] * (padded // block_size))

    if len(sorted_ids_list) == 0:
        sorted_ids = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
        expert_ids = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
        num_tokens_post_padded = torch.tensor(0, dtype=torch.int32, device=topk_ids.device)
        return sorted_ids, expert_ids, num_tokens_post_padded

    sorted_ids = torch.tensor(sorted_ids_list, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_padded = torch.tensor(sorted_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, num_tokens_post_padded


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    """
    Zero out the output tile for masked or filtered tokens for the current N-tile.

    Writes zeros into `c_ptr` at rows given by `offs_token` (subject to `token_mask`)
    and columns for the tile `pid_n`.
    """
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
):
    """
    Fused MoE matmul tile kernel: A(EM, K) x B(E, K, N) -> C(EM, N).

    Processes tokens in a pre-grouped, padded order (`sorted_token_ids`), where
    each BLOCK_SIZE_M row tile belongs to an expert block (`expert_ids`). For each
    MxN tile:
      - Map rows back to original tokens (and their experts) using routing buffers
      - Load A rows for those tokens and the B slice for the current expert
      - Accumulate dot products over K in BLOCK_SIZE_K chunks
      - Optionally multiply by per-token routed weight
      - Store results at the token rows in C

    Many parameters are present for feature parity but are dormant in this demo
    (e.g., quantization and bias). Compute type is bf16 or fp16.
    """
    # Compute tile indices in the logical MxN grid
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Early-exit if this M-tile begins past the padded token range
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Gather the token indices for this tile and build a mask for valid tokens
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)
    if filter_expert and off_experts == -1:
        # If this block is filtered out, write zeros to output and exit
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # Column offsets for the N tile and K offsets for the inner loop
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Build pointers into A (token rows) and B (expert slice) for the current tile
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not even_Ks:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        # Load A and B tiles and accumulate the dot product
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        b = b.to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        # Apply per-token routed weights to the output tile if requested
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    # Store the result back to the rows corresponding to the current tokens
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    top_k: int,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 8,
    mul_routed_weight: bool = False,
):
    """
    Host-side launcher for `fused_moe_kernel`.

    Computes grid and meta-parameters from input tensor shapes, selects compute
    precision from A's dtype, and dispatches the Triton kernel with the routing
    buffers and tiling configuration.
    """
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2]
    even_Ks = (K % BLOCK_SIZE_K) == 0
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        A,
        None,
        B,
        None,
        None,
        C,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        K,
        sorted_token_ids.shape[0],
        top_k * (A.shape[0]),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        0,
        0,
        C.stride(0),
        C.stride(1),
        0,
        0,
        0,
        0,
        0,
        group_n=0,
        group_k=0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=even_Ks,
        c_sorted=False,
        filter_expert=False,
    )


@torch.inference_mode()
def run_demo(
    *,
    m: int = 8,
    k: int = 64,
    d_ff: int = 128,
    num_experts: int = 4,
    top_k: int = 2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a self-contained fused MoE demo: up -> SiLU*mul -> down.

    Constructs random inputs and expert weights, computes top-k routing,
    aligns token-expert pairs into kernel-friendly blocks, runs two expert
    matmuls with a gated activation in between, scatters results back, and
    compares the Triton output to a PyTorch reference implementation.

    Returns:
        out_triton: (M, K) tensor from the Triton fused path.
        out_ref:    (M, K) tensor from the PyTorch reference path.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton fused MoE demo.")
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_states = torch.randn(m, k, device=device, dtype=dtype)
    w1 = torch.randn(num_experts, 2 * d_ff, k, device=device, dtype=dtype) / math.sqrt(k)
    w2 = torch.randn(num_experts, k, d_ff, device=device, dtype=dtype) / math.sqrt(d_ff)

    router_logits = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = _topk_router(hidden_states, router_logits, top_k, True)

    # Prepare sorted/padded token ordering
    BLOCK_SIZE_M = 64
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        topk_ids.to(torch.int32), BLOCK_SIZE_M, num_experts
    )

    EM = int(num_tokens_post_padded.item())
    if EM == 0:
        return hidden_states, hidden_states

    # GEMM-1 (Up+Gate): A(M,K) x B1(E, N1, K) -> C1(EM, N1)
    N1 = 2 * d_ff
    c1 = torch.empty((EM, N1), device=device, dtype=dtype)
    _invoke_fused_moe_kernel(
        hidden_states,
        w1,
        c1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        mul_routed_weight=False,
    )

    # Activation: SiLU * mul
    inter = torch.empty((EM, d_ff), device=device, dtype=dtype)
    inter.copy_(_silu_and_mul(c1))

    # GEMM-2 (Down): A(EM, d_ff) x B2(E, K, d_ff) -> C2(EM, K)
    c2 = torch.empty((EM, k), device=device, dtype=dtype)
    _invoke_fused_moe_kernel(
        inter,
        w2.transpose(1, 2).contiguous(),  # shape (E, d_ff, K) -> want (E, K, d_ff): keep B as (E, N, K)
        c2,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        mul_routed_weight=False,
    )

    # Scatter back to (m, top_k, k), then weight-sum
    sentinel = m * top_k
    valid = sorted_token_ids < sentinel
    flat_ids = sorted_token_ids[valid].to(torch.long)
    m_idx = flat_ids // top_k
    j_idx = flat_ids % top_k
    Z2 = torch.zeros((m * top_k, k), device=device, dtype=dtype)
    lin = m_idx * top_k + j_idx
    Z2.index_add_(0, lin, c2[valid])
    Z = Z2.view(m, top_k, k)
    out_triton = (Z * topk_weights.to(dtype).unsqueeze(-1)).sum(dim=1)

    # Reference
    out_ref = torch.empty_like(out_triton)
    for i in range(m):
        x_i = hidden_states[i]
        acc = torch.zeros(k, device=device, dtype=dtype)
        for j in range(top_k):
            e = int(topk_ids[i, j].item())
            a_ij = topk_weights[i, j].to(dtype)
            gate_up_i = F.linear(x_i, w1[e])
            inter_i = _silu_and_mul(gate_up_i)
            y_e = F.linear(inter_i, w2[e])
            acc = acc + a_ij * y_e
        out_ref[i] = acc

    return out_triton, out_ref


def main():
    """
    Entry point: execute the demo and print validation statistics.
    """
    try:
        out_vec, out_ref = run_demo()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    diff = (out_vec - out_ref).abs().max().item()
    print("Self-contained Triton fused MoE (up → SiLU*mul → down)")
    print(f"max_abs_diff = {diff:.6e}")
    print(f"output dtype/shape: {out_vec.dtype} {tuple(out_vec.shape)}")


if __name__ == "__main__":
    main()




