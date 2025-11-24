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
        self.create_weights(torch.bfloat16) # create weights as bf16, later cast to fp8 if needed
        
        # fp8-specific initializations
        if self.use_deep_gemm_fp8:
            if dg is None:
                raise ImportError("deep_gemm is not available!")
            self.prepare_deep_gemm_fp8_weights()
            # pre-allocate fixed size input buffers
            self.fp8_up_in_buf = torch.empty(
                num_experts,
                max_batch_size,
                self.hidden_size,
                device=self.w13_weight.device,
                dtype=torch.float8_e4m3fn,
            )
            self.fp8_down_in_buf = torch.empty(
                num_experts,
                max_batch_size,
                self.intermediate_size,
                device=self.w2_weight.device,
                dtype=torch.float8_e4m3fn,
            )
            # pre-allocate scale buffers: one scale per 128-wide block along K
            # DeepGEMM masked kernel and quant functions require scale buffers to have this alignment
            up_scale_blocks = dg.ceil_div(self.hidden_size, 128)
            down_scale_blocks = dg.ceil_div(self.intermediate_size, 128)
            self.fp8_up_scale_buf = torch.empty(
                num_experts,
                max_batch_size,
                up_scale_blocks,
                device=self.w13_weight.device,
                dtype=torch.float32,
            )
            self.fp8_down_scale_buf = torch.empty(
                num_experts,
                max_batch_size,
                down_scale_blocks,
                device=self.w2_weight.device,
                dtype=torch.float32,
            )
            # create cached grouped gemm buffers
            self.create_grouped_gemm_cache(params_dtype, enable_cutlass_cache, max_batch_size)
            # change the views
            self.cache_up = self.cache_up.view(self.num_experts, -1, self.intermediate_size * 2)
            self.cache_down = self.cache_down.view(self.num_experts, -1, self.hidden_size)
            
            # capture the cudagraph for deep_gemm forward pass
            self.graph = torch.cuda.CUDAGraph()
            # Note that bs is a scalar to the kernel, so it has to be baked in, thus we use a conservative upper bound
            self.static_bs = self.num_experts * max_batch_size # conservative upper bound
            
            # We need static 'batch_sizes' tensor for the mask, later update the contents in each forward pass
            self.static_batch_sizes = torch.zeros(self.num_experts, dtype=torch.int32, device="cuda")
            
            # Warmup
            with torch.no_grad():
                self._forward_deep_gemm_internal()
                torch.cuda.synchronize()
            
            # Capture
            with torch.cuda.graph(self.graph):
                self._forward_deep_gemm_internal()
        else:
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

    def prepare_deep_gemm_fp8_weights(self):
        # We assume the weights are already initialized in BF16 in self.w13_weight and self.w2_weight
        # We need to transpose them and cast to FP8 for deep_gemm
        # w13_weight: [E, H, I*2] -> [E, I*2, H] for deep_gemm (if NT, B is transposed)
        # w2_weight: [E, I, H] -> [E, H, I]
        
        # Scaling factors logic: this doesn't affect performance, and weights are 
        # usually pre-quantized, so we just use ones for now.
        
        # w13
        self.w13_weight_fp8 = self.w13_weight.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
        # For w13: K = hidden_size, N = intermediate_size * 2
        k_w13 = self.hidden_size
        n_w13 = self.intermediate_size * 2
        # Each weight scale entry corresponds to a 128-wide tile along N and K.
        # dg.ceil_div(dim, 128) gives the number of such tiles needed to cover that dimension.
        self.w13_sf = torch.ones(
            self.num_experts, 
            dg.ceil_div(n_w13, 128), 
            dg.ceil_div(k_w13, 128), 
            device=self.w13_weight.device, 
            dtype=torch.float32
        )
        
        # w2
        self.w2_weight_fp8 = self.w2_weight.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
        # For w2: K = intermediate_size, N = hidden_size
        k_w2 = self.intermediate_size
        n_w2 = self.hidden_size
        self.w2_sf = torch.ones(
            self.num_experts, 
            dg.ceil_div(n_w2, 128), 
            dg.ceil_div(k_w2, 128), 
            device=self.w2_weight.device, 
            dtype=torch.float32
        )

    def create_grouped_gemm_cache(self, params_dtype, enable_cutlass_cache, max_batch_size):
        # TODO: for now we interpret max_batch_size as per-expert batch size
        # but this leads to wasted memory for non-masked grouped gemm path, which will only use the first max_batch_size tokens
        # later we should 1) differentiate between per-expert and per-rank batch sizes 2) support cudagraph for non-masked grouped gemm path
        total_capacity = self.num_experts * max_batch_size
        self.cache_up = torch.empty((total_capacity, self.intermediate_size * 2), dtype=params_dtype, device=torch.device("cuda"))
        self.cache_down = torch.empty((total_capacity, self.hidden_size), dtype=params_dtype, device=torch.device("cuda"))
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
        # 1. Prepare Inputs outside graph
        
        # Cast hiddens to FP8 (Dynamic shape, cannot be in graph)
        hiddens_fp8, sf_hiddens = dg.per_token_cast_to_fp8(hiddens, use_ue8m0=False)
        
        # Scatter to fixed input buffe
        start = 0
        batch_sizes_cpu = batch_sizes.cpu() # this should already be on CPU, just make sure here
        for i in range(self.num_experts):
            length = batch_sizes_cpu[i].item()
            if length > 0:
                self.fp8_up_in_buf[i, :length].copy_(hiddens_fp8[start : start + length])
                self.fp8_up_scale_buf[i, :length].copy_(sf_hiddens[start : start + length])
                start += length

        # Update static mask
        self.static_batch_sizes.copy_(batch_sizes)

        # 2. Replay Graph (Compute)
        self.graph.replay()
        
        # 3. Gather Output (Dynamic shape, cannot be in graph)
        # Construct packed output [bs, hidden]
        final_out = torch.empty(bs, self.hidden_size, dtype=torch.bfloat16, device=hiddens.device)
        
        start = 0
        for i in range(self.num_experts):
            length = batch_sizes_cpu[i].item()
            if length > 0:
                final_out[start : start + length].copy_(
                    self.cache_down[i, :length]
                )
                start += length
                
        return final_out

    def _forward_deep_gemm_internal(self):
        # Everything here uses FIXED shapes and FIXED pointers
        
        # Run w13 kernel
        dg.m_grouped_fp8_gemm_nt_masked(
            (self.fp8_up_in_buf, self.fp8_up_scale_buf),
            (self.w13_weight_fp8, self.w13_sf),
            self.cache_up, #output buffer
            self.static_batch_sizes,
            self.static_bs # baked-in capacity
        )
        
        # Activation and gating
        # In-place modification of cache_up is fine
        # We perform this on the WHOLE buffer (including padding) to keep shape static
        # Logic: up = SiLU(gate) * val
        self.act_fn(self.cache_up[:, :, :self.intermediate_size]) # in-place SiLU on gate
        up_res = self.cache_up[:, :, :self.intermediate_size] * self.cache_up[:, :, self.intermediate_size:]
        
        # TODO: can we merge the below quant + copy into a single kernel?
        # Quantize + Copy to Down Input
        # per_token_cast_to_fp8 expects [M, K] and returns:
        #   up_fp8_flat: [M, K]
        #   up_res_sf_flat: [M, ceil_div(K, 128)]
        # Here M = num_experts * max_batch_size, K = intermediate_size
        up_res_flat = up_res.view(-1, self.intermediate_size)
        up_fp8_flat, up_res_sf_flat = dg.per_token_cast_to_fp8(up_res_flat, use_ue8m0=False)

        # Reshape back to [E, BS, K] and [E, BS, ceil_div(K,128)]
        max_bs = self.fp8_down_in_buf.shape[1]
        up_fp8 = up_fp8_flat.view(self.num_experts, max_bs, self.intermediate_size)
        n_scale_down = dg.ceil_div(self.intermediate_size, 128)
        up_res_sf = up_res_sf_flat.view(self.num_experts, max_bs, n_scale_down)

        self.fp8_down_in_buf.copy_(up_fp8)
        self.fp8_down_scale_buf.copy_(up_res_sf)
        
        # Run w2 kernel
        dg.m_grouped_fp8_gemm_nt_masked(
            (self.fp8_down_in_buf, self.fp8_down_scale_buf),
            (self.w2_weight_fp8, self.w2_sf),
            self.cache_down,
            self.static_batch_sizes,
            self.static_bs
        )

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