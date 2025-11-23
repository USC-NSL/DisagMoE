import torch
from typing import override, List, Optional
from grouped_gemm.backend import gmm
from disagmoe.utils.constants import MAX_BATCH_SIZE
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from disagmoe.models.linear import ReplicatedLinear

# Optional import for deep_gemm (only available for sm90+)
try:
    import deep_gemm as dg
except ImportError:
    dg = None

class MoEExperts(torch.nn.Module):
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int,
        num_experts: int, 
        tp_size: int = 1,
        enable_cutlass_cache: bool = True,
        max_batch_size: int = MAX_BATCH_SIZE,
        use_deep_gemm_fp8: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        self.use_deep_gemm_fp8 = use_deep_gemm_fp8
        assert tp_size == 1, "Not implemented TP for experts yet"
            
        params_dtype = torch.get_default_dtype()
        assert params_dtype == torch.bfloat16, "Only bf16 is supported for now"
        self.create_weights(params_dtype)
        
        if self.use_deep_gemm_fp8:
            if dg is None:
                raise ImportError("deep_gemm is not available!")
            self.prepare_deep_gemm_weights()
            
        self.gmm_with_cache = None
        self.gmm = gmm
        self.gmm_cache_max_batch_size = max_batch_size
        self.create_grouped_gemm_cache(params_dtype, enable_cutlass_cache, max_batch_size)
        
    def create_weights(self, params_dtype: torch.dtype):
        self.w13_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.hidden_size,
                                                    self.intermediate_size * 2,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w13_weight", self.w13_weight)
        
        self.w2_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.intermediate_size,
                                                    self.hidden_size,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w2_weight", self.w2_weight)
        
        self.act_fn = torch.nn.SiLU(inplace=True)

    def prepare_deep_gemm_weights(self):
        # We assume the weights are already initialized in BF16 in self.w13_weight and self.w2_weight
        # We need to transpose them and cast to FP8 for deep_gemm
        # w13_weight: [E, H, I*2] -> [E, I*2, H] for deep_gemm (if NT, B is transposed)
        # w2_weight: [E, I, H] -> [E, H, I]
        
        # Scaling factors logic: this doesn't affect performance, and weights are 
        # usually pre-quantized, so we just use ones for now.
        # ceil_div = lambda x, y: (x + y - 1) // y
        def ceil_div(x, y): return (x + y - 1) // y
        
        # w13
        self.w13_weight_fp8 = self.w13_weight.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
        # sfa is dynamic (input), sfb is static (weight)
        # sfb shape: [G, ceil_div(N, 128), ceil_div(K, 128)]
        # For w13: K = hidden_size, N = intermediate_size * 2
        k_w13 = self.hidden_size
        n_w13 = self.intermediate_size * 2
        self.w13_sfb = torch.ones(
            self.num_experts, 
            ceil_div(n_w13, 128), 
            ceil_div(k_w13, 128), 
            device=self.w13_weight.device, 
            dtype=torch.float32
        )
        
        # w2
        self.w2_weight_fp8 = self.w2_weight.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
        # For w2: K = intermediate_size, N = hidden_size
        k_w2 = self.intermediate_size
        n_w2 = self.hidden_size
        self.w2_sfb = torch.ones(
            self.num_experts, 
            ceil_div(n_w2, 128), 
            ceil_div(k_w2, 128), 
            device=self.w2_weight.device, 
            dtype=torch.float32
        )

    def create_grouped_gemm_cache(self, params_dtype, enable_cutlass_cache, max_batch_size):
        self.cache_up = torch.empty((max_batch_size, self.intermediate_size * 2), dtype=params_dtype, device=torch.device("cuda"))
        self.cache_down = torch.empty((max_batch_size, self.hidden_size), dtype=params_dtype, device=torch.device("cuda"))
        if enable_cutlass_cache:
            from grouped_gemm.backend import get_arguments, gmm_with_arguments

            self.cutlass_workspace_size, self.arguments_ptr = get_arguments(
                self.num_experts, torch.device("cuda"))
            self.cutlass_workspace = torch.empty(
                [self.cutlass_workspace_size], dtype=torch.uint8, device=torch.device("cuda"))

            def _gmm(hiddens, weight, batch_sizes, **kwargs):
                return gmm_with_arguments(hiddens, weight, batch_sizes, self.cutlass_workspace, self.arguments_ptr, **kwargs)
            
            self.gmm_with_cache = _gmm
        
    def forward(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        if self.use_deep_gemm_fp8:
            return self._forward_deep_gemm(bs, hiddens, batch_sizes)
        
        output = None
        if bs < self.gmm_cache_max_batch_size and self.gmm_with_cache is not None:
            up = self.gmm_with_cache(hiddens, self.w13_weight, batch_sizes, c=self.cache_up)
            up = self.act_fn(up[:bs, :self.intermediate_size]) * up[:bs, self.intermediate_size:]
            down = self.gmm_with_cache(up, self.w2_weight, batch_sizes, c=self.cache_down)
            output = down[:bs]
        else:
            up = self.gmm(hiddens, self.w13_weight, batch_sizes)
            up = self.act_fn(up[:, :self.intermediate_size]) * up[:, self.intermediate_size:]
            down = self.gmm(up, self.w2_weight, batch_sizes)
            output = down
        return output

    def _forward_deep_gemm(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        
        # 1. Prepare inputs for w13
        # hiddens: [M, K] -> convert to FP8
        # m_indices: generate from batch_sizes
        
        # Generate m_indices
        # batch_sizes is [num_experts], containing count of tokens per expert
        expert_ids = torch.arange(self.num_experts, device=hiddens.device, dtype=torch.int32)
        m_indices = torch.repeat_interleave(expert_ids, batch_sizes.to(device=hiddens.device, dtype=torch.int32))
        
        # Cast hiddens to FP8 using deep_gemm utility
        hiddens_fp8, sfa_hiddens = dg.per_token_cast_to_fp8(hiddens, use_ue8m0=False)
        
        # Output buffer for w13 (BF16)
        # shape: [M, intermediate_size * 2]
        M = hiddens.shape[0]
        intermediate_size_2 = self.intermediate_size * 2
        up_out = torch.empty(M, intermediate_size_2, device=hiddens.device, dtype=torch.bfloat16)
        
        # Run w13 kernel
        # w13_weight_fp8: [E, I*2, H] (transposed)
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (hiddens_fp8, sfa_hiddens), 
            (self.w13_weight_fp8, self.w13_sfb), 
            up_out, 
            m_indices
        )
        
        # Activation and gating
        # up_out is [M, intermediate_size * 2]
        # Split into gate and value
        up = self.act_fn(up_out[:, :self.intermediate_size]) * up_out[:, self.intermediate_size:]
        
        # 2. Prepare inputs for w2
        # up: [M, intermediate_size] -> convert to FP8
        up_fp8, sfa_up = dg.per_token_cast_to_fp8(up, use_ue8m0=False)
        
        # Output buffer for w2 (BF16)
        # shape: [M, hidden_size]
        down_out = torch.empty(M, self.hidden_size, device=hiddens.device, dtype=torch.bfloat16)
        
        # Run w2 kernel
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (up_fp8, sfa_up), 
            (self.w2_weight_fp8, self.w2_sfb), 
            down_out, 
            m_indices
        )
        
        return down_out

class MoEExpertsSerial(MoEExperts):
    
    def __init__(self, hidden_size, intermediate_size, num_experts, tp_size = 1, 
                 max_batch_size: int = MAX_BATCH_SIZE,
                 quant_config: Optional[QuantizationConfig] = None):
        # Store quantization config before parent ctor calls create_weights
        self._moe_quant_config: Optional[QuantizationConfig] = quant_config
        super().__init__(hidden_size, intermediate_size, num_experts, tp_size, enable_cutlass_cache=False)
    
    @override
    def create_weights(self, params_dtype: torch.dtype):
        # Only override if we want to quantize MoE layers
        if getattr(self, "_moe_quant_config", None) is None:
            return super().create_weights(params_dtype)
        else:
            self.act_fn = torch.nn.SiLU(inplace=True)
            self.up_linears = torch.nn.ModuleList([
                ReplicatedLinear(
                    input_size=self.hidden_size,
                    output_size=self.intermediate_size * 2,
                    bias=False,
                    params_dtype=params_dtype,
                    quant_config=self._moe_quant_config,
                ).cuda() for _ in range(self.num_experts)
            ])
            self.down_linears = torch.nn.ModuleList([
                ReplicatedLinear(
                    input_size=self.intermediate_size,
                    output_size=self.hidden_size,
                    bias=False,
                    params_dtype=params_dtype,
                    quant_config=self._moe_quant_config,
                ).cuda() for _ in range(self.num_experts)
            ])
            return
        
    @override
    def forward(self, num_tokens: int, hiddens: torch.Tensor, batch_sizes: List[int]):
        
        def calc(input, local_expert_id: int):
            # Quantized path using vLLM Linear wrappers
            if getattr(self, "_moe_quant_config", None) is not None:
                up, _ = self.up_linears[local_expert_id](input)
                up = self.act_fn(up[:, :self.intermediate_size]) * up[:, self.intermediate_size:]
                down, _ = self.down_linears[local_expert_id](up)
                return down
            else:
                up = torch.matmul(input, self.w13_weight[local_expert_id])
                up = self.act_fn(up[:, :self.intermediate_size]) * up[:, self.intermediate_size:]
                down = torch.matmul(up, self.w2_weight[local_expert_id])
                return down
        
        if len(batch_sizes) == 1:
            return calc(hiddens, 0)
        
        s = 0
        results = []
        for i, bs in enumerate(batch_sizes):
            if bs == 0:
                continue
            cur_hiddens = hiddens[s: s + bs]
            results.append(calc(cur_hiddens, i))
            s += bs
        
        if len(results) == 1:
            return results[0]
        return torch.cat(results)