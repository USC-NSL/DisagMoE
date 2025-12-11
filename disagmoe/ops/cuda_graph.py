

import torch
from torch import Tensor

import triton.language as tl
import triton

@triton.jit
def cuda_graph_preprocess_triton(
    # Destination pointers
    static_input_ptr, static_positions_ptr, static_slot_mapping_ptr,
    static_block_table_ptr, static_seq_lens_ptr, static_context_lens_ptr,
    static_seq_start_loc_ptr,
    # Source pointers
    hidden_states_ptr, positions_ptr, slot_mapping_ptr,
    block_tables_ptr, seq_lens_tensor_ptr, context_lens_tensor_ptr,
    seq_start_loc_ptr,
    # Dimensions
    num_tokens, hidden_dim, max_num_blocks,
    # Strides
    static_input_stride, hidden_states_stride,
    static_block_table_stride_0, static_block_table_stride_1,
    block_tables_stride_0, block_tables_stride_1,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    HIDDEN_BLOCK_SIZE: tl.constexpr,
):
    # For hidden_states (2D tensor)
    pid_token = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    # Calculate offsets for hidden states (this is more complex)
    token_offset = pid_token * TOKEN_BLOCK_SIZE
    hidden_offset = pid_hidden * HIDDEN_BLOCK_SIZE
    
    # Block for hidden_states copy
    if pid_hidden < (hidden_dim + HIDDEN_BLOCK_SIZE - 1) // HIDDEN_BLOCK_SIZE:
        # Get token indices
        token_indices = token_offset + tl.arange(0, TOKEN_BLOCK_SIZE)
        hidden_indices = hidden_offset + tl.arange(0, HIDDEN_BLOCK_SIZE)
        
        # Create masks for boundary checking
        token_mask = token_indices < num_tokens
        hidden_mask = hidden_indices < hidden_dim
        
        # Load from hidden_states
        # The offset calculation is different for 2D tensors
        offsets_hidden = (token_indices[:, None] * hidden_states_stride + hidden_indices[None, :])
        hidden_vals = tl.load(hidden_states_ptr + offsets_hidden, mask=token_mask[:, None] & hidden_mask[None, :])
        
        # Store to static_input
        offsets_static = (token_indices[:, None] * static_input_stride + hidden_indices[None, :])
        tl.store(static_input_ptr + offsets_static, hidden_vals, mask=token_mask[:, None] & hidden_mask[None, :])
    
    # For 1D tensors (positions, slot_mapping, seq_lens, context_lens)
    if pid_hidden == 0:  # Only need one block in the hidden dimension
        token_indices = token_offset + tl.arange(0, TOKEN_BLOCK_SIZE)
        mask = token_indices < num_tokens
        
        # Copy positions
        positions_vals = tl.load(positions_ptr + token_indices, mask=mask)
        tl.store(static_positions_ptr + token_indices, positions_vals, mask=mask)
        
        # Copy slot_mapping
        slot_mapping_vals = tl.load(slot_mapping_ptr + token_indices, mask=mask)
        tl.store(static_slot_mapping_ptr + token_indices, slot_mapping_vals, mask=mask)
        
        # Copy seq_lens
        seq_lens_vals = tl.load(seq_lens_tensor_ptr + token_indices, mask=mask)
        tl.store(static_seq_lens_ptr + token_indices, seq_lens_vals, mask=mask)
        
        # Copy context_lens
        context_lens_vals = tl.load(context_lens_tensor_ptr + token_indices, mask=mask)
        tl.store(static_context_lens_ptr + token_indices, context_lens_vals, mask=mask)
    
    # Special handling for seq_start_loc (size is num_tokens + 1)
    if pid_hidden == 0:
        token_indices = token_offset + tl.arange(0, TOKEN_BLOCK_SIZE)
        mask = token_indices < (num_tokens + 1)  # +1 for seq_start_loc
        
        seq_start_vals = tl.load(seq_start_loc_ptr + token_indices, mask=mask)
        tl.store(static_seq_start_loc_ptr + token_indices, seq_start_vals, mask=mask)
    
    # For block_tables (2D tensor)
    if pid_hidden < (max_num_blocks + HIDDEN_BLOCK_SIZE - 1) // HIDDEN_BLOCK_SIZE:
        token_indices = token_offset + tl.arange(0, TOKEN_BLOCK_SIZE)
        block_indices = pid_hidden * HIDDEN_BLOCK_SIZE + tl.arange(0, HIDDEN_BLOCK_SIZE)
        token_mask = token_indices < num_tokens
        block_mask = block_indices < max_num_blocks
        
        # Load from block_tables
        block_vals = tl.load(
            block_tables_ptr + token_indices[:, None] * block_tables_stride_0 + block_indices[None, :] * block_tables_stride_1, 
            mask=token_mask[:, None] & block_mask[None, :]
        )
        
        # Store to static_block_table
        tl.store(
            static_block_table_ptr + token_indices[:, None] * static_block_table_stride_0 + block_indices[None, :] * static_block_table_stride_1,
            block_vals, 
            mask=token_mask[:, None] & block_mask[None, :]
        )

@triton.jit
def cuda_graph_preprocess_triton_opt(
    # Inputs
    hidden_states_ptr,      # [T, H]
    block_tables_ptr,       # [T, B]
    positions_ptr,          # [T]
    slot_mapping_ptr,       # [T]
    seq_lens_ptr,           # [T]
    context_lens_ptr,       # [T]
    seq_start_loc_ptr,      # [T+1]

    # Outputs
    out_hidden_ptr,         # [T, H]
    out_block_tables_ptr,   # [T, B]
    out_positions_ptr,      # [T]
    out_slot_mapping_ptr,   # [T]
    out_seq_lens_ptr,       # [T]
    out_context_lens_ptr,   # [T]
    out_seq_start_loc_ptr,  # [T+1]

    T: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # -------------------------------------------------------
    # Case 1: per-token block copying hidden + block_tables
    # -------------------------------------------------------
    if pid < T:
        t = pid

        # Offsets for row t
        offs = tl.arange(0, BLOCK)

        # ---- Copy hidden_states[t, :] ----
        for h0 in range(0, H, BLOCK):
            idx = h0 + offs
            mask = idx < H
            val = tl.load(hidden_states_ptr + t*H + idx, mask=mask)
            tl.store(out_hidden_ptr + t*H + idx, val, mask=mask)

        # ---- Copy block_tables[t, :] ----
        for b0 in range(0, B, BLOCK):
            idx = b0 + offs
            mask = idx < B
            val = tl.load(block_tables_ptr + t*B + idx, mask=mask)
            tl.store(out_block_tables_ptr + t*B + idx, val, mask=mask)

        return

    # -------------------------------------------------------
    # Case 2: one dedicated block handles all 1-D tensors
    # -------------------------------------------------------
    if pid == T:
        offs = tl.arange(0, BLOCK)

        # positions
        for i0 in range(0, T, BLOCK):
            idx = i0 + offs
            mask = idx < T
            val = tl.load(positions_ptr + idx, mask=mask)
            tl.store(out_positions_ptr + idx, val, mask=mask)

        # slot mapping
        for i0 in range(0, T, BLOCK):
            idx = i0 + offs
            mask = idx < T
            val = tl.load(slot_mapping_ptr + idx, mask=mask)
            tl.store(out_slot_mapping_ptr + idx, val, mask=mask)

        # seq_lens
        for i0 in range(0, T, BLOCK):
            idx = i0 + offs
            mask = idx < T
            val = tl.load(seq_lens_ptr + idx, mask=mask)
            tl.store(out_seq_lens_ptr + idx, val, mask=mask)

        # context_lens
        for i0 in range(0, T, BLOCK):
            idx = i0 + offs
            mask = idx < T
            val = tl.load(context_lens_ptr + idx, mask=mask)
            tl.store(out_context_lens_ptr + idx, val, mask=mask)

        # seq_start_loc (T+1)
        for i0 in range(0, T+1, BLOCK):
            idx = i0 + offs
            mask = idx < (T+1)
            val = tl.load(seq_start_loc_ptr + idx, mask=mask)
            tl.store(out_seq_start_loc_ptr + idx, val, mask=mask)

def cuda_graph_preprocess_cuda(
    hidden_states: Tensor,
    positions: Tensor,
    block_tables: Tensor,
    slot_mapping: Tensor,
    seq_lens_tensor: Tensor,
    context_lens_tensor: Tensor,
    seq_start_loc: Tensor,
    
    static_input: Tensor,
    static_positions: Tensor,
    static_block_table: Tensor,
    static_slot_mapping: Tensor,
    static_seq_lens: Tensor,
    static_context_lens: Tensor,
    static_seq_start_loc: Tensor,
    
    padded_batch_size: int,
    tokens_per_block: int = 2,
):
    torch.ops.disag_ops.cuda_graph_preprocess_fused(
        hidden_states,
        positions,
        block_tables,
        slot_mapping,
        seq_lens_tensor,
        context_lens_tensor,
        seq_start_loc,

        static_input,
        static_positions,
        static_block_table,
        static_slot_mapping,
        static_seq_lens,
        static_context_lens,
        static_seq_start_loc,
        padded_batch_size,
        tokens_per_block,
    )
    
def copy_graph_results_cuda(
    tokens: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    out_tokens: Tensor,
    out_topk_ids: Tensor,
    out_topk_weights: Tensor,
    num_tokens: int = 0,
):
    torch.ops.disag_ops.copy_graph_results_fused(
        tokens, topk_ids, topk_weights, out_tokens, out_topk_ids, out_topk_weights, num_tokens)
    
def fused_copy_and_pad_cuda(
    hidden_states: Tensor,
    batch_sizes: Tensor,
    m_indices: Tensor,
    out_hiddens: Tensor,
    out_batch_sizes: Tensor,
    out_m_indices: Tensor,
    padded_bsz: int,
    tokens_per_block: int = 2,
):
    torch.ops.disag_ops.fused_copy_and_pad(
        hidden_states,
        batch_sizes,
        m_indices,
        out_hiddens,
        out_batch_sizes,
        out_m_indices,
        padded_bsz,
        tokens_per_block,
    )