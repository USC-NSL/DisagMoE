import torch
from typing import override, List, Optional, Dict
from grouped_gemm.backend import gmm
from disagmoe.utils.constants import MAX_BATCH_SIZE
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from disagmoe.models.linear import ReplicatedLinear
from disagmoe.models.quantization import sglang_per_token_group_quant_fp8
from disagmoe.utils.logger import get_logger
from disagmoe.ops.cuda_graph import fused_copy_and_pad_cuda

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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        assert tp_size == 1, "Not implemented TP for experts yet"
            
        params_dtype = torch.get_default_dtype()
        assert params_dtype == torch.bfloat16, "Only bf16 is supported for now"
        # create weights as bf16
        self.create_weights(torch.bfloat16)

        # grouped_gemm (bf16) path
        self.gmm_with_cache = None
        self.gmm = gmm
        self.gmm_cache_max_batch_size = max_batch_size
        self.create_grouped_gemm_cache(params_dtype, enable_cutlass_cache, max_batch_size)

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

    def create_grouped_gemm_cache(self, params_dtype, enable_cutlass_cache, max_batch_size):
        # TODO: for now we interpret max_batch_size as per-expert batch size
        # but this leads to wasted memory for non-masked grouped gemm path, which will only use the first max_batch_size tokens
        # later we should 1) differentiate between per-expert and per-rank batch sizes 2) support cudagraph for non-masked grouped gemm path
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
        if enable_cutlass_cache:
            from grouped_gemm.backend import get_arguments, gmm_with_arguments

            self.cutlass_workspace_size, self.arguments_ptr = get_arguments(
                self.num_experts, torch.device("cuda")
            )
            self.cutlass_workspace = torch.empty(
                [self.cutlass_workspace_size],
                dtype=torch.uint8,
                device=torch.device("cuda"),
            )

            def _gmm(hiddens, weight, batch_sizes, **kwargs):
                return gmm_with_arguments(
                    hiddens,
                    weight,
                    batch_sizes,
                    self.cutlass_workspace,
                    self.arguments_ptr,
                    **kwargs,
                )

            self.gmm_with_cache = _gmm

    def forward(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        output = None
        if bs < self.gmm_cache_max_batch_size and self.gmm_with_cache is not None:
            up = self.gmm_with_cache(hiddens, self.w13_weight, batch_sizes, c=self.cache_up)
            up = self.act_fn(up[:bs, : self.intermediate_size]) * up[:bs, self.intermediate_size :]
            down = self.gmm_with_cache(up, self.w2_weight, batch_sizes, c=self.cache_down)
            output = down[:bs]
        else:
            up = self.gmm(hiddens, self.w13_weight, batch_sizes)
            up = self.act_fn(up[:, : self.intermediate_size]) * up[:, self.intermediate_size :]
            down = self.gmm(up, self.w2_weight, batch_sizes)
            output = down
        return output


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
        self.expert_ids = torch.arange(self.num_experts, device="cpu", dtype=torch.int32)

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
        self.w13_weight_fp8 = torch.nn.Parameter(w13_init_bf16.to(torch.float8_e4m3fn), requires_grad=False)
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
        self.w2_weight_fp8 = torch.nn.Parameter(w2_init_bf16.to(torch.float8_e4m3fn), requires_grad=False)
        # For w2: K = intermediate_size, N = hidden_size
        self.w2_sf = torch.ones(
            self.num_experts,
            dg.ceil_div(n_w2, 128),
            dg.ceil_div(k_w2, 128),
            device=self.w2_weight_fp8.device,
            dtype=torch.float32,
        )

    def forward(self, bs: int, hiddens: torch.Tensor, m_indices: torch.Tensor):
        # Cast hiddens to FP8
        # For sglang with DeepEP, the cast is fused with communication.
        hiddens_fp8, sf_hiddens = sglang_per_token_group_quant_fp8(
            hiddens, group_size=128, scale_ue8m0=False
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
        up = self.act_fn(up_out[:, : self.intermediate_size]) * up_out[
            :, self.intermediate_size :
        ]

        # Prepare inputs for w2
        up_fp8, sf_up = sglang_per_token_group_quant_fp8(
            up, group_size=128, scale_ue8m0=False
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


class MoEExpertsDeepGemmFP8Graph(MoEExpertsDeepGemmFP8):
    """DeepGEMM-based FP8 grouped experts using CUDAGraphs with bucketing."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        tp_size: int = 1,
        max_batch_size: int = MAX_BATCH_SIZE,
    ):
        super().__init__(hidden_size, intermediate_size, num_experts, tp_size, max_batch_size)

        self.graph_batch_sizes = self.get_graph_batch_sizes(max_batch_size)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Warmup and capture
        self.capture_graphs()

    def get_graph_batch_sizes(self, graph_max_batch_size: int) -> List[int]:
        """
        We use 128-aligned, exponentially growing buckets:
        128, 256, 512, 1024, 2048, ... up to graph_max_batch_size.
        """
        if graph_max_batch_size <= 0:
            raise ValueError("graph_max_batch_size must be positive")

        # If max batch size is small, just capture exactly that size.
        if graph_max_batch_size <= 128:
            return [graph_max_batch_size]

        graph_bsz: List[int] = []
        bsz = 128
        while bsz < graph_max_batch_size:
            graph_bsz.append(bsz)
            bsz *= 2

        # Always include the exact max batch size as the last bucket
        if graph_bsz[-1] != graph_max_batch_size:
            graph_bsz.append(graph_max_batch_size)

        return graph_bsz

    def _get_graph_by_batch_size(self, batch_size: int):
        for bs in self.graph_batch_sizes:
            if bs >= batch_size:
                return bs
        raise RuntimeError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")

    def capture_graphs(self):
        # We need to capture a graph for each bucket size
        for bs in self.graph_batch_sizes:
            self.static_buffers[bs] = self._allocate_static_buffers(bs)
            
            # Warmup
            self._run_graph_pass(bs, warmup=True)
            torch.cuda.synchronize()
            
            # Capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._run_graph_pass(bs, warmup=False)
            self.graphs[bs] = g

    def _allocate_static_buffers(self, bs: int) -> Dict[str, torch.Tensor]:
        buffers: Dict[str, torch.Tensor] = {}
        device = self.w13_weight_fp8.device
        
        # Input hiddens
        buffers["hiddens"] = torch.zeros((bs, self.hidden_size), dtype=torch.bfloat16, device=device)
        
        # m_indices
        buffers["m_indices"] = torch.zeros((bs,), dtype=torch.int32, device=device)
        
        # Output: w2 output (BF16)
        buffers["down_out"] = torch.empty((bs, self.hidden_size), dtype=torch.bfloat16, device=device)
        
        return buffers

    def _run_graph_pass(self, bs: int, warmup: bool = False):
        buffers = self.static_buffers[bs]
        
        hiddens_fp8 = torch.empty(
            (bs, self.hidden_size),
            dtype=torch.float8_e4m3fn,
            device=buffers["hiddens"].device,
        )
        n_scales_hiddens = dg.ceil_div(self.hidden_size, 128)
        hiddens_sf = torch.empty(
            (bs, n_scales_hiddens),
            dtype=torch.float32,
            device=buffers["hiddens"].device,
        )

        up_out = torch.empty(
            (bs, self.intermediate_size * 2),
            dtype=torch.bfloat16,
            device=buffers["hiddens"].device,
        )

        up_activated = torch.empty(
            (bs, self.intermediate_size),
            dtype=torch.bfloat16,
            device=buffers["hiddens"].device,
        )

        up_fp8 = torch.empty(
            (bs, self.intermediate_size),
            dtype=torch.float8_e4m3fn,
            device=buffers["hiddens"].device,
        )
        n_scales_up = dg.ceil_div(self.intermediate_size, 128)
        up_sf = torch.empty(
            (bs, n_scales_up),
            dtype=torch.float32,
            device=buffers["hiddens"].device,
        )

        # 1. Quantize Hiddens
        sglang_per_token_group_quant_fp8(
            buffers["hiddens"],
            group_size=128,
            scale_ue8m0=False,
            out_q=hiddens_fp8,
            out_s=hiddens_sf,
        )
        
        # 2. GEMM w13
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (hiddens_fp8, hiddens_sf),
            (self.w13_weight_fp8, self.w13_sf),
            up_out,
            buffers["m_indices"],
        )
        
        # 3. Activation
        gate = up_out[:, : self.intermediate_size]
        val = up_out[:, self.intermediate_size :]
        
        # In-place SiLU on gate
        self.act_fn(gate) 
        
        # Element-wise mul. `gate * val` -> up_activated
        torch.mul(gate, val, out=up_activated)

        # 4. Quantize Up
        sglang_per_token_group_quant_fp8(
            up_activated,
            group_size=128,
            scale_ue8m0=False,
            out_q=up_fp8,
            out_s=up_sf,
        )

        # 5. GEMM w2
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (up_fp8, up_sf),
            (self.w2_weight_fp8, self.w2_sf),
            buffers["down_out"],
            buffers["m_indices"],
        )
        
    @override
    def forward(self, bs: int, hiddens: torch.Tensor, m_indices: torch.Tensor):
        # 1. Select bucket
        bucket_bs = self._get_graph_by_batch_size(bs)
        buffers = self.static_buffers[bucket_bs]
        
        # 2. Fused Copy + Pad (CUDA Op)
        fused_copy_and_pad_cuda(
            hiddens, m_indices,
            buffers["hiddens"], buffers["m_indices"],
            bucket_bs,
        )

        # # Data copy integrity check for -1 case
        # m_idx = buffers["m_indices"].cpu()
        # neg_indices = (m_idx == -1).nonzero(as_tuple=True)[0]
        # if len(neg_indices) > 0:
        #     first_neg = neg_indices[0].item()
        #     if not torch.all(m_idx[first_neg:] == -1).item():
        #         raise RuntimeError(
        #             f"m_indices check failed: Found non -1 values after the first -1. "
        #             f"bs={bs}, bucket_bs={bucket_bs}, m_indices.dtype={m_indices.dtype}. "
        #             f"Buffer: {m_idx.tolist()} Original m_indices: {m_indices.tolist()}"
        #         )
        #     valid_m_idx = m_idx[:first_neg]
        # else:
        #     valid_m_idx = m_idx
        #
        # if valid_m_idx.numel() > 0:
        #     if torch.any(valid_m_idx < 0).item():
        #         raise RuntimeError(
        #             f"m_indices check failed: Found negative values in valid part. "
        #             f"bs={bs}, bucket_bs={bucket_bs}, m_indices.dtype={m_indices.dtype}. "
        #             f"Buffer: {m_idx.tolist()} Original m_indices: {m_indices.tolist()}"
        #         )
        #     if torch.any(valid_m_idx >= self.num_experts).item():
        #         raise RuntimeError(
        #             f"m_indices check failed: Found values >= num_experts ({self.num_experts}). "
        #             f"bs={bs}, bucket_bs={bucket_bs}, m_indices.dtype={m_indices.dtype}. "
        #             f"Buffer: {m_idx.tolist()} Original m_indices: {m_indices.tolist()}"
        #         )
        #     if torch.any(valid_m_idx[1:] < valid_m_idx[:-1]).item():
        #         raise RuntimeError(
        #             f"m_indices check failed: Valid part is not monotonically increasing. "
        #             f"bs={bs}, bucket_bs={bucket_bs}, m_indices.dtype={m_indices.dtype}. "
        #             f"Buffer: {m_idx.tolist()} Original m_indices: {m_indices.tolist()}"
        #         )

        
        # # a simpler check for 0 case
        # with torch.no_grad():
        #     src_idx = m_indices.to("cpu")
        #     dst_idx = buffers["m_indices"][: src_idx.numel()].to("cpu")
        #     mismatches = (src_idx != dst_idx).nonzero(as_tuple=True)[0]
        #     if mismatches.numel() > 0:
        #         first_bad = mismatches[0].item()
        #         raise RuntimeError(
        #             "m_indices copy check failed: destination buffer does not match source "
        #             f"at position {first_bad}. "
        #             f"bs={bs}, bucket_bs={bucket_bs}, "
        #             f"src_val={int(src_idx[first_bad])}, "
        #             f"dst_val={int(dst_idx[first_bad])}, "
        #             f"src={src_idx.tolist()}, "
        #             f"dst={dst_idx.tolist()}"
        #         )

        # 3. Replay Graph
        self.graphs[bucket_bs].replay()
        
        # 4. Return output sliced
        return buffers["down_out"][:bs]


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
        enable_cutlass_cache: bool = True,
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

        # create cached grouped gemm buffers (bf16 cache)
        cache_dtype = torch.bfloat16
        self.create_grouped_gemm_cache(cache_dtype, enable_cutlass_cache, max_batch_size)
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
        self.w13_weight_fp8 = torch.nn.Parameter(w13_init_bf16.to(torch.float8_e4m3fn), requires_grad=False)
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
        self.w2_weight_fp8 = torch.nn.Parameter(w2_init_bf16.to(torch.float8_e4m3fn), requires_grad=False)
        self.w2_sf = torch.ones(
            self.num_experts, 
            dg.ceil_div(n_w2, 128), 
            dg.ceil_div(k_w2, 128), 
            device=self.w2_weight_fp8.device,
            dtype=torch.float32
        )

    def create_grouped_gemm_cache(self, params_dtype, enable_cutlass_cache, max_batch_size):
        total_capacity = self.num_experts * max_batch_size
        self.cache_up = torch.empty((total_capacity, self.intermediate_size * 2), dtype=params_dtype, device=torch.device("cuda"))
        self.cache_down = torch.empty((total_capacity, self.hidden_size), dtype=params_dtype, device=torch.device("cuda"))
        if enable_cutlass_cache:
            from grouped_gemm.backend import get_arguments, gmm_with_arguments

            self.cutlass_workspace_size, self.arguments_ptr = get_arguments(
                self.num_experts, torch.device("cuda"))
            self.cutlass_workspace = torch.empty([self.cutlass_workspace_size], dtype=torch.uint8, device=torch.device("cuda"))

            def _gmm(hiddens, weight, batch_sizes, **kwargs):
                return gmm_with_arguments(hiddens, weight, batch_sizes, self.cutlass_workspace, self.arguments_ptr, **kwargs)
            
            self.gmm_with_cache = _gmm
        
    def forward(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        # Cast hiddens to FP8 (Dynamic shape, cannot be in graph)
        hiddens_fp8, sf_hiddens = dg.per_token_cast_to_fp8(hiddens, use_ue8m0=False)
        
        # Scatter to fixed input buffer
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

        # Replay Graph (Compute)
        self.graph.replay()
        
        # Gather Output (Dynamic shape, cannot be in graph)
        # Construct packed output [bs, hidden]
        final_out = torch.empty(bs, self.hidden_size, dtype=torch.bfloat16, device=hiddens.device)
        
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
        
        s = 0
        results = []
        for i, bs in enumerate(batch_sizes):
            if bs == 0:
                continue
            cur_hiddens = hiddens[s: s + bs]
            results.append(calc(cur_hiddens, i))
            s += bs
            
        return torch.cat(results)