from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import MixtralConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from disagmoe.models.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from disagmoe.ops.memory import  permute_tokens_cuda

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk)

import triton
import triton.language as tl


@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert = tl.program_id(0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert + 1, target_location + 1)

@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)


class MoEAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int,
        top_k: int = 1,
        tp_size: int = 1,
        tp_rank: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_experts = num_experts
        self.top_k = top_k
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        
        # NOTE(shaoyuw): must invoke initialize_model_parallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            params_dtype=params_dtype,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            params_dtype=params_dtype,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)
        
        # Gate always runs at half / full precision for now.

        self.gate = ReplicatedLinear(hidden_size,
                                     num_experts,
                                     bias=False,
                                     params_dtype=params_dtype,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        
        self.pre_attention_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def permute_by_exp_ids(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Permute the attention output by expert IDs.
        """
        _, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)

        permuted_output = permute_tokens_cuda(hidden_states, reorder_ids)
        
        return permuted_output, reorder_ids
    
    def random_routing(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random routing.
        """
        batch_size = hidden_states.size(0)
        expert_logits = torch.rand(batch_size, self.num_experts, device=hidden_states.device)
        topk_weights, topk_ids = torch.topk(expert_logits, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_attention_layernorm(hidden_states)
        else:
            hidden_states, residual = self.pre_attention_layernorm(hidden_states, residual)
        
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        output, residual = self.post_attention_layernorm(output, residual)
        router_logits, _ = self.gate(output)

        # print(f"hidden_states {hidden_states} output {output}, router logits {router_logits}")
        
        topk_weights, topk_ids = fused_topk(hidden_states=hidden_states,
                                gating_output=router_logits,
                                topk=self.top_k,
                                renormalize=True)
        
        # print(f"topk info {topk_weights}, {topk_ids}")
        
        topk_weights, topk_ids = self.random_routing(hidden_states)

        permuted_output, reorder_ids = self.permute_by_exp_ids(output, topk_ids)

        return permuted_output, topk_weights, topk_ids, reorder_ids