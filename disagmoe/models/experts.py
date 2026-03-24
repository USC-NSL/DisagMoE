import torch
from typing import override, List, Optional, Dict
from disagmoe.utils.constants import MAX_BATCH_SIZE
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from disagmoe.models.linear import ReplicatedLinear
from disagmoe.ops.quantization import sglang_per_token_group_quant_fp8
from disagmoe.utils.logger import get_logger

import disagmoe_c

# Optional import for deep_gemm (only available for sm90+)
try:
    import deep_gemm as dg
except ImportError:
    dg = None


class MoEExpertsCUTLASS(torch.nn.Module):
    """CUTLASS Grouped-GEMM MoE experts for sm < 90.

    Each instance owns two CutlassGemmRunner objects (one for w13, one for w2).
    forward() calls runner.setup_meta() + runner.run() which are both
    graph-capturable (static launch configs, fixed buffer addresses).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        tp_size: int = 1,
        max_batch_size: int = MAX_BATCH_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        self.max_batch_size = max_batch_size
        assert tp_size == 1, "Not implemented TP for experts yet"

        params_dtype = torch.get_default_dtype()
        assert params_dtype == torch.bfloat16, "Only bf16 is supported for now"
        self.create_weights(torch.bfloat16)

        # One-time hardware probe + tile selection
        from disagmoe.ops.grouped_gemm import ensure_initialized

        ensure_initialized()
        # Create per-weight GEMM runners (no global map, explicit ownership)
        self.w13_runner = disagmoe_c.CutlassGemmRunner(self.w13_weight, max_batch_size)
        self.w2_runner = disagmoe_c.CutlassGemmRunner(self.w2_weight, max_batch_size)

    def create_weights(self, params_dtype: torch.dtype):
        self.w13_weight = torch.nn.Parameter(
            torch.randn(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size * 2,
                dtype=params_dtype,
            ).cuda(),
            requires_grad=False,
        )
        self.register_parameter("w13_weight", self.w13_weight)

        self.w2_weight = torch.nn.Parameter(
            torch.randn(
                self.num_experts,
                self.intermediate_size,
                self.hidden_size,
                dtype=params_dtype,
            ).cuda(),
            requires_grad=False,
        )
        self.register_parameter("w2_weight", self.w2_weight)

        self.act_fn = torch.nn.SiLU(inplace=True)

    def forward(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        # During graph capture, these allocations are redirected to graph's memory pool
        cache_up = torch.empty(
            (bs, self.intermediate_size * 2),
            dtype=torch.bfloat16,
            device=hiddens.device,
        )
        down_out = torch.empty(
            (bs, self.hidden_size),
            dtype=torch.bfloat16,
            device=hiddens.device,
        )

        # Up projection: [tokens, hidden] @ [E, hidden, inter*2] -> [tokens, inter*2]
        self.w13_runner.setup_meta(hiddens, cache_up, batch_sizes)
        self.w13_runner.run()
        up = (
            self.act_fn(cache_up[:, : self.intermediate_size])
            * cache_up[:, self.intermediate_size :]
        )

        # Down projection: [tokens, inter] @ [E, inter, hidden] -> [tokens, hidden]
        self.w2_runner.setup_meta(up, down_out, batch_sizes)
        self.w2_runner.run()
        return down_out


class MoEExpertsCUTLASSFP8(torch.nn.Module):
    """W8A16 FP8-quantized CUTLASS grouped-GEMM MoE experts for sm < 90.

    Weights are stored permanently as ``float8_e4m3fn`` with per-channel
    ``float32`` scale factors ``[E, N]``, following the same convention as
    ``MoEExpertsDeepGemmFP8``.  The C++ ``CutlassGemmRunnerFP8`` owns a
    BF16 workspace and runs a fused dequant kernel every ``setup_meta()``
    call (graph-capturable), so no BF16 weight copy is kept in Python.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        tp_size: int = 1,
        max_batch_size: int = MAX_BATCH_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        self.max_batch_size = max_batch_size
        assert tp_size == 1, "Not implemented TP for experts yet"

        self.create_weights()

        from disagmoe.ops.grouped_gemm import ensure_initialized

        ensure_initialized()

        self.w13_runner = disagmoe_c.CutlassGemmRunnerFP8(
            self.w13_weight, self.w13_weight_scale, max_batch_size
        )
        self.w2_runner = disagmoe_c.CutlassGemmRunnerFP8(
            self.w2_weight, self.w2_weight_scale, max_batch_size
        )

    def create_weights(self):
        E = self.num_experts
        K13, N13 = self.hidden_size, self.intermediate_size * 2
        K2, N2 = self.intermediate_size, self.hidden_size

        self.act_fn = torch.nn.SiLU(inplace=True)

        w13_init = torch.randn(E, K13, N13, device="cuda", dtype=torch.bfloat16)
        self.w13_weight = torch.nn.Parameter(
            w13_init.to(torch.float8_e4m3fn), requires_grad=False
        )
        self.register_buffer(
            "w13_weight_scale", torch.ones(E, N13, device="cuda", dtype=torch.float32)
        )

        w2_init = torch.randn(E, K2, N2, device="cuda", dtype=torch.bfloat16)
        self.w2_weight = torch.nn.Parameter(
            w2_init.to(torch.float8_e4m3fn), requires_grad=False
        )
        self.register_buffer(
            "w2_weight_scale", torch.ones(E, N2, device="cuda", dtype=torch.float32)
        )

    def forward(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        cache_up = torch.empty(
            (bs, self.intermediate_size * 2),
            dtype=torch.bfloat16,
            device=hiddens.device,
        )
        down_out = torch.empty(
            (bs, self.hidden_size),
            dtype=torch.bfloat16,
            device=hiddens.device,
        )

        # Quantize BF16 activations to FP8 on Python side (following DeepGEMM path)
        # Use group_size that divides hidden dim (2880 % 128 != 0, but 2880 % 64 == 0)
        gs_w13 = 128 if hiddens.shape[-1] % 128 == 0 else 64
        hiddens_fp8, _sf = sglang_per_token_group_quant_fp8(
            hiddens, group_size=gs_w13, scale_ue8m0=False
        )

        self.w13_runner.setup_meta(hiddens_fp8, cache_up, batch_sizes)
        self.w13_runner.run()
        up = (
            self.act_fn(cache_up[:, : self.intermediate_size])
            * cache_up[:, self.intermediate_size :]
        )

        # Quantize intermediate activations to FP8
        gs_w2 = 128 if up.shape[-1] % 128 == 0 else 64
        up_fp8, _sf_up = sglang_per_token_group_quant_fp8(
            up, group_size=gs_w2, scale_ue8m0=False
        )

        self.w2_runner.setup_meta(up_fp8, down_out, batch_sizes)
        self.w2_runner.run()
        return down_out


class MoEExpertsDeepGemmBF16(torch.nn.Module):
    """DeepGEMM-based BF16 grouped experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        tp_size: int = 1,
        max_batch_size: int = MAX_BATCH_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        self.max_batch_size = max_batch_size
        assert tp_size == 1, "Not implemented TP for experts yet"

        if dg is None:
            raise ImportError("deep_gemm is not available!")
        # Initialize weights directly in the layout expected by DeepGEMM kernels.
        self.create_weights()

        self.expert_ids = torch.arange(
            self.num_experts, device="cpu", dtype=torch.int32
        )

    def create_weights(self):
        self.act_fn = torch.nn.SiLU(inplace=True)

        # For w13: K = hidden_size, N = intermediate_size * 2
        k_w13 = self.hidden_size
        n_w13 = self.intermediate_size * 2
        # DeepGEMM expects weights shaped [E, N, K] with per-[128x128] block scales.
        self.w13 = torch.randn(
            self.num_experts,
            n_w13,
            k_w13,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # w2
        # For w2: K = intermediate_size, N = hidden_size
        k_w2 = self.intermediate_size
        n_w2 = self.hidden_size
        self.w2 = torch.randn(
            self.num_experts,
            n_w2,
            k_w2,
            device="cuda",
            dtype=torch.bfloat16,
        )

    def forward(self, bs: int, hiddens: torch.Tensor, m_indices: torch.Tensor):
        # Output buffer for w13 (BF16), shape: [total_tokens, intermediate_size * 2]
        intermediate_size_2 = self.intermediate_size * 2
        up_out = torch.empty(
            bs,
            intermediate_size_2,
            device=hiddens.device,
            dtype=torch.bfloat16,
        )

        # Run w13 kernel (non-masked, contiguous)
        dg.m_grouped_bf16_gemm_nt_contiguous(
            hiddens,
            self.w13,
            up_out,
            m_indices,
        )

        # Activation and gating
        up = (
            self.act_fn(up_out[:, : self.intermediate_size])
            * up_out[:, self.intermediate_size :]
        )

        # Output buffer for w2 (BF16), shape: [M, hidden_size]
        down_out = torch.empty(
            bs,
            self.hidden_size,
            device=hiddens.device,
            dtype=torch.bfloat16,
        )

        # Run w2 kernel (non-masked, contiguous)
        dg.m_grouped_bf16_gemm_nt_contiguous(
            up,
            self.w2,
            down_out,
            m_indices,
        )

        return down_out


class MoEExpertsDeepGemmFP8(torch.nn.Module):
    """DeepGEMM-based FP8 grouped experts, legacy non-graph, non-masked path."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        tp_size: int = 1,
        max_batch_size: int = MAX_BATCH_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        self.max_batch_size = max_batch_size
        assert tp_size == 1, "Not implemented TP for experts yet"

        if dg is None:
            raise ImportError("deep_gemm is not available!")
        # Initialize weights directly in the layout expected by DeepGEMM kernels.
        self.create_weights()

        # init for non-masked, non-graph deep_gemm path
        self.expert_ids = torch.arange(
            self.num_experts, device="cpu", dtype=torch.int32
        )

    def create_weights(self):
        """Allocate FP8 weights directly in the DeepGEMM-preferred layout."""

        self.act_fn = torch.nn.SiLU(inplace=True)

        # For w13: K = hidden_size, N = intermediate_size * 2
        k_w13 = self.hidden_size
        n_w13 = self.intermediate_size * 2
        # DeepGEMM expects weights shaped [E, N, K] with per-[128x128] block scales.
        w13_init_bf16 = torch.randn(
            self.num_experts,
            n_w13,
            k_w13,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.w13_weight_fp8 = torch.nn.Parameter(
            w13_init_bf16.to(torch.float8_e4m3fn), requires_grad=False
        )
        # For w13: K = hidden_size, N = intermediate_size * 2
        # Each weight scale entry corresponds to a 128-wide tile along N and K.
        # dg.ceil_div(dim, 128) gives the number of such tiles needed to cover that dimension.
        self.w13_sf = torch.ones(
            self.num_experts,
            dg.ceil_div(n_w13, 128),
            dg.ceil_div(k_w13, 128),
            device=self.w13_weight_fp8.device,
            dtype=torch.float32,
        )

        # w2
        # For w2: K = intermediate_size, N = hidden_size
        k_w2 = self.intermediate_size
        n_w2 = self.hidden_size
        w2_init_bf16 = torch.randn(
            self.num_experts,
            n_w2,
            k_w2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.w2_weight_fp8 = torch.nn.Parameter(
            w2_init_bf16.to(torch.float8_e4m3fn), requires_grad=False
        )
        # For w2: K = intermediate_size, N = hidden_size
        self.w2_sf = torch.ones(
            self.num_experts,
            dg.ceil_div(n_w2, 128),
            dg.ceil_div(k_w2, 128),
            device=self.w2_weight_fp8.device,
            dtype=torch.float32,
        )

    def forward(self, bs: int, hiddens: torch.Tensor, m_indices: torch.Tensor):
        # Quant input
        # For sglang with DeepEP, the cast is fused with communication.
        gs_w13 = 128 if hiddens.shape[-1] % 128 == 0 else 64
        hiddens_fp8, sf_hiddens = sglang_per_token_group_quant_fp8(
            hiddens, group_size=gs_w13, scale_ue8m0=False
        )

        # Output buffer for w13 (BF16), shape: [total_tokens, intermediate_size * 2]
        intermediate_size_2 = self.intermediate_size * 2
        up_out = torch.empty(
            bs,
            intermediate_size_2,
            device=hiddens.device,
            dtype=torch.bfloat16,
        )

        # Run w13 kernel (non-masked, contiguous)
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (hiddens_fp8, sf_hiddens),
            (self.w13_weight_fp8, self.w13_sf),
            up_out,
            m_indices,
        )

        # Activation and gating
        up = (
            self.act_fn(up_out[:, : self.intermediate_size])
            * up_out[:, self.intermediate_size :]
        )

        # Quant
        gs_w2 = 128 if up.shape[-1] % 128 == 0 else 64
        up_fp8, sf_up = sglang_per_token_group_quant_fp8(
            up, group_size=gs_w2, scale_ue8m0=False
        )

        # Output buffer for w2 (BF16), shape: [M, hidden_size]
        down_out = torch.empty(
            bs,
            self.hidden_size,
            device=hiddens.device,
            dtype=torch.bfloat16,
        )

        # Run w2 kernel (non-masked, contiguous)
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (up_fp8, sf_up),
            (self.w2_weight_fp8, self.w2_sf),
            down_out,
            m_indices,
        )

        return down_out


class MoEExpertsDeepGemmFP8Masked(torch.nn.Module):
    """DeepGEMM-based FP8 grouped experts using masked kernels + CUDA graphs.

    Note: This is WIP and currently not wired into the executor.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        tp_size: int = 1,
        max_batch_size: int = MAX_BATCH_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        self.max_batch_size = max_batch_size
        assert tp_size == 1, "Not implemented TP for experts yet"

        if dg is None:
            raise ImportError("deep_gemm is not available!")
        # Initialize weights directly in the layout expected by DeepGEMM kernels.
        self.create_weights()

        # pre-allocate fixed size input buffers
        self.fp8_up_in_buf = torch.empty(
            num_experts,
            max_batch_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        self.fp8_down_in_buf = torch.empty(
            num_experts,
            max_batch_size,
            self.intermediate_size,
            device="cuda",
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
            device="cuda",
            dtype=torch.float32,
        )
        self.fp8_down_scale_buf = torch.empty(
            num_experts,
            max_batch_size,
            down_scale_blocks,
            device="cuda",
            dtype=torch.float32,
        )

        # create cached output buffers (bf16 cache)
        cache_dtype = torch.bfloat16
        self._create_cache_buffers(cache_dtype, max_batch_size)
        # change the views
        self.cache_up = self.cache_up.view(
            self.num_experts, -1, self.intermediate_size * 2
        )
        self.cache_down = self.cache_down.view(self.num_experts, -1, self.hidden_size)

        # capture the cudagraph for deep_gemm forward pass
        self.graph = torch.cuda.CUDAGraph()
        # Note that bs is a scalar to the kernel, so it has to be baked in, thus we use a conservative upper bound
        self.static_bs = self.num_experts * max_batch_size  # conservative upper bound

        # We need static 'batch_sizes' tensor for the mask, later update the contents in each forward pass
        self.static_batch_sizes = torch.zeros(
            self.num_experts, dtype=torch.int32, device="cuda"
        )

        # Warmup
        with torch.no_grad():
            self._forward_deep_gemm_internal()
            torch.cuda.synchronize()

        # Capture
        with torch.cuda.graph(self.graph):
            self._forward_deep_gemm_internal()

    def create_weights(self):
        """Allocate FP8 weights directly in the DeepGEMM-preferred layout."""

        self.act_fn = torch.nn.SiLU(inplace=True)

        # w13: K = hidden_size, N = intermediate_size * 2
        k_w13 = self.hidden_size
        n_w13 = self.intermediate_size * 2
        w13_init_bf16 = torch.randn(
            self.num_experts,
            n_w13,
            k_w13,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.w13_weight_fp8 = torch.nn.Parameter(
            w13_init_bf16.to(torch.float8_e4m3fn), requires_grad=False
        )
        self.w13_sf = torch.ones(
            self.num_experts,
            dg.ceil_div(n_w13, 128),
            dg.ceil_div(k_w13, 128),
            device=self.w13_weight_fp8.device,
            dtype=torch.float32,
        )

        # w2
        k_w2 = self.intermediate_size
        n_w2 = self.hidden_size
        w2_init_bf16 = torch.randn(
            self.num_experts,
            n_w2,
            k_w2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.w2_weight_fp8 = torch.nn.Parameter(
            w2_init_bf16.to(torch.float8_e4m3fn), requires_grad=False
        )
        self.w2_sf = torch.ones(
            self.num_experts,
            dg.ceil_div(n_w2, 128),
            dg.ceil_div(k_w2, 128),
            device=self.w2_weight_fp8.device,
            dtype=torch.float32,
        )

    def _create_cache_buffers(self, params_dtype, max_batch_size):
        total_capacity = self.num_experts * max_batch_size
        self.cache_up = torch.empty(
            (total_capacity, self.intermediate_size * 2),
            dtype=params_dtype,
            device=torch.device("cuda"),
        )
        self.cache_down = torch.empty(
            (total_capacity, self.hidden_size),
            dtype=params_dtype,
            device=torch.device("cuda"),
        )

    def forward(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        # Cast hiddens to FP8 (Dynamic shape, cannot be in graph)
        hiddens_fp8, sf_hiddens = dg.per_token_cast_to_fp8(hiddens, use_ue8m0=False)

        # Scatter to fixed input buffer
        start = 0
        batch_sizes_cpu = (
            batch_sizes.cpu()
        )  # this should already be on CPU, just make sure here
        for i in range(self.num_experts):
            length = batch_sizes_cpu[i].item()
            if length > 0:
                self.fp8_up_in_buf[i, :length].copy_(
                    hiddens_fp8[start : start + length]
                )
                self.fp8_up_scale_buf[i, :length].copy_(
                    sf_hiddens[start : start + length]
                )
                start += length

        # Update static mask
        self.static_batch_sizes.copy_(batch_sizes)

        # Replay Graph (Compute)
        self.graph.replay()

        # Gather Output (Dynamic shape, cannot be in graph)
        # Construct packed output [bs, hidden]
        final_out = torch.empty(
            bs, self.hidden_size, dtype=torch.bfloat16, device=hiddens.device
        )

        start = 0
        for i in range(self.num_experts):
            length = batch_sizes_cpu[i].item()
            if length > 0:
                final_out[start : start + length].copy_(self.cache_down[i, :length])
                start += length

        return final_out

    def _forward_deep_gemm_internal(self):
        # Everything here uses FIXED shapes and FIXED pointers

        # Run w13 kernel
        dg.m_grouped_fp8_gemm_nt_masked(
            (self.fp8_up_in_buf, self.fp8_up_scale_buf),
            (self.w13_weight_fp8, self.w13_sf),
            self.cache_up,  # output buffer
            self.static_batch_sizes,
            self.static_bs,  # baked-in capacity
        )

        # Activation and gating
        # In-place modification of cache_up is fine
        # We perform this on the WHOLE buffer (including padding) to keep shape static
        # Logic: up = SiLU(gate) * val
        self.act_fn(
            self.cache_up[:, :, : self.intermediate_size]
        )  # in-place SiLU on gate
        up_res = (
            self.cache_up[:, :, : self.intermediate_size]
            * self.cache_up[:, :, self.intermediate_size :]
        )

        # TODO: can we merge the below quant + copy into a single kernel?
        # Quantize + Copy to Down Input
        # per_token_cast_to_fp8 expects [M, K] and returns:
        #   up_fp8_flat: [M, K]
        #   up_res_sf_flat: [M, ceil_div(K, 128)]
        # Here M = num_experts * max_batch_size, K = intermediate_size
        up_res_flat = up_res.view(-1, self.intermediate_size)
        up_fp8_flat, up_res_sf_flat = dg.per_token_cast_to_fp8(
            up_res_flat, use_ue8m0=False
        )

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
            self.static_bs,
        )


class MoEExpertsSerial(MoEExpertsCUTLASS):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts,
        tp_size=1,
        max_batch_size: int = MAX_BATCH_SIZE,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        # Store quantization config before parent ctor calls create_weights
        self._moe_quant_config: Optional[QuantizationConfig] = quant_config
        super().__init__(hidden_size, intermediate_size, num_experts, tp_size)

    @override
    def create_weights(self, params_dtype: torch.dtype):
        # Only override if we want to quantize MoE layers
        if getattr(self, "_moe_quant_config", None) is None:
            return super().create_weights(params_dtype)
        else:
            self.act_fn = torch.nn.SiLU(inplace=True)
            self.up_linears = torch.nn.ModuleList(
                [
                    ReplicatedLinear(
                        input_size=self.hidden_size,
                        output_size=self.intermediate_size * 2,
                        bias=False,
                        params_dtype=params_dtype,
                        quant_config=self._moe_quant_config,
                    ).cuda()
                    for _ in range(self.num_experts)
                ]
            )
            self.down_linears = torch.nn.ModuleList(
                [
                    ReplicatedLinear(
                        input_size=self.intermediate_size,
                        output_size=self.hidden_size,
                        bias=False,
                        params_dtype=params_dtype,
                        quant_config=self._moe_quant_config,
                    ).cuda()
                    for _ in range(self.num_experts)
                ]
            )
            return

    @override
    def forward(self, num_tokens: int, hiddens: torch.Tensor, batch_sizes: List[int]):
        def calc(input, local_expert_id: int):
            # Quantized path using vLLM Linear wrappers
            if getattr(self, "_moe_quant_config", None) is not None:
                up, _ = self.up_linears[local_expert_id](input)
                up = (
                    self.act_fn(up[:, : self.intermediate_size])
                    * up[:, self.intermediate_size :]
                )
                down, _ = self.down_linears[local_expert_id](up)
                return down
            else:
                up = torch.matmul(input, self.w13_weight[local_expert_id])
                up = (
                    self.act_fn(up[:, : self.intermediate_size])
                    * up[:, self.intermediate_size :]
                )
                down = torch.matmul(up, self.w2_weight[local_expert_id])
                return down

        s = 0
        results = []
        for i, bs in enumerate(batch_sizes):
            if bs == 0:
                continue
            cur_hiddens = hiddens[s : s + bs]
            results.append(calc(cur_hiddens, i))
            s += bs

        return torch.cat(results)


class SharedExpertMLP(torch.nn.Module):
    """Shared expert MLP that processes ALL tokens (no routing).

    Runs on the attention GPU after post-attention layernorm.
    Uses ReplicatedLinear for automatic FP8 quantization support
    when quant_config is provided.

    Architecture: gate_up_proj -> SiLU gate -> down_proj
    Same structure as a single routed expert but applied to every token.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.gate_up_proj = ReplicatedLinear(
            input_size=hidden_size,
            output_size=intermediate_size * 2,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = ReplicatedLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gate_up: [num_tokens, hidden_size] -> [num_tokens, intermediate_size * 2]
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate = gate_up[:, : self.intermediate_size]
        up = gate_up[:, self.intermediate_size :]
        # SiLU-gated activation
        x = self.act_fn(gate) * up
        # down: [num_tokens, intermediate_size] -> [num_tokens, hidden_size]
        down, _ = self.down_proj(x)
        return down
