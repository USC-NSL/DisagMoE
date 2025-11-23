import torch
import matplotlib.pyplot as plt
import numpy as np
from grouped_gemm.backend import gmm
import os
import csv
from triton_moe_demo import (
    _moe_align_block_size,
    _invoke_fused_moe_kernel,
    _silu_and_mul,
)

try:
    import deep_gemm as dg
    HAS_DEEP_GEMM = True
except ImportError:
    HAS_DEEP_GEMM = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.set_device(device)
torch.set_default_device(device)
# Dtype configuration: bf16 (default), fp16, or fp8
_DTYPE_STR = os.environ.get("GGEMM_DTYPE", "bf16").lower()
if _DTYPE_STR == "fp16":
    DTYPE = torch.float16
elif _DTYPE_STR == "bf16":
    DTYPE = torch.bfloat16
elif _DTYPE_STR == "fp8":
    # Use bfloat16 as default for non-fp8 tensors (e.g. activations before cast)
    DTYPE = torch.bfloat16
else:
    raise ValueError(f"Unsupported GGEMM_DTYPE '{_DTYPE_STR}'. Use one of: bf16, fp16, fp8.")

torch.set_default_dtype(DTYPE)
print(f"GGEMM dtype: {DTYPE}")

# Minimal Perfetto trace enablement via env var PERFETTO_TRACE
trace_path = os.environ.get("PERFETTO_TRACE")
_prof = None
if trace_path:
    _prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    )
    _prof.start()

@torch.inference_mode()
def alloc_expert_weights(
    hidden_size,
    intermediate_size,
    num_experts,
    device,
    dtype=DTYPE,
):
    BCs = torch.randn(
        num_experts, hidden_size, intermediate_size * 2, device=device, dtype=dtype
    ).contiguous()
    Ds = torch.randn(
        num_experts, intermediate_size, hidden_size, device=device, dtype=dtype
    ).contiguous()
    return BCs, Ds

 

@torch.inference_mode()
def benchmark_grouped_gemm(hidden_size, intermediate_size, num_experts, label):

    # this is just batch sizes we test
    # row_sizes = np.concatenate((np.arange(4, 128, 4), np.arange(128, 512 + 1, 32)))
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512, 1024])

    num_repeats = 5

    BCs, Ds = alloc_expert_weights(hidden_size, intermediate_size, num_experts, device, dtype=DTYPE)

    # --- Sequntial CUDA graph case ---

    # GLU here, not SwiGLU; computational load are roughly the same
    def run_expert(x, BC, D, up_buf, down_buf):
        torch.matmul(x, BC, out=up_buf)
        up1 = up_buf[:, :intermediate_size]
        up3 = up_buf[:, intermediate_size:]
        up1 *= up3
        _ = torch.matmul(up1, D, out=down_buf)

    results_sequntial = []

    for n_rows in row_sizes:
        results_this_batch_size = []
        # per-expert variable batch sizes
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]
        up_bufs = [torch.empty((bs, intermediate_size * 2), device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]
        down_bufs = [torch.empty((bs, hidden_size), device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]

        # warm-up
        for _ in range(2):
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])
        
        # capture cuda graph
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])
        torch.cuda.synchronize(device)

        # warm-up graph
        for _ in range(2):
            graph.replay()
        
        # actual measurments
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_sequntial.append(np.mean(results_this_batch_size))

    # --- sequntial non-CUDAGraph case

    results_sequntial_no_graph = []

    for n_rows in row_sizes:
        results_this_batch_size = []
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]
        up_bufs = [torch.empty((bs, intermediate_size * 2), device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]
        down_bufs = [torch.empty((bs, hidden_size), device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]
        
        # warm-up
        for _ in range(2):
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])
        
        # actual measurements
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for i in range(num_experts):
                run_expert(As_list[i], BCs[i], Ds[i], up_bufs[i], down_bufs[i])
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_sequntial_no_graph.append(np.mean(results_this_batch_size))
    
    # --- grouped GEMM case ---

    # Note that here we always make sure the up/down buffers exactly match
    # the total batch size. In the case the total tokens is less than
    # the buffer sizes, remember to truncate the output tensors of gmm.
    def run_experts_grouped_gemm(As_list, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes):
        A_flat = torch.cat(As_list, dim=0).contiguous()
        up = gmm(A_flat, BCs, batch_sizes, c=up_buf)
        # NOTE: the up_1 slice is NOT contiguous
        # Compute GLU into a new contiguous buffer to satisfy gmm's contiguity requirement
        up_1 = up[:, :intermediate_size]
        up_3 = up[:, intermediate_size:]
        glu = (up_1 * up_3).contiguous()
        _ = gmm(glu, Ds, batch_sizes, c=down_buf)
    
    results_grouped_gemm = []

    for n_rows in row_sizes:
        results_this_batch_size = []

        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]
        total_rows = int(sum(batch_sizes_i))
        # torch.empty gives contiguous buffers by default
        up_buf = torch.empty((total_rows, intermediate_size * 2), device=device, dtype=DTYPE)
        down_buf = torch.empty((total_rows, hidden_size), device=device, dtype=DTYPE)
        batch_sizes = torch.tensor(batch_sizes_i, device=device, dtype=torch.int64)

        # warm-up
        for _ in range(2):
            run_experts_grouped_gemm(As_list, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes)
        
        # capture cuda graph
        stream =  torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            run_experts_grouped_gemm(As_list, BCs, Ds, intermediate_size, up_buf, down_buf, batch_sizes)
        torch.cuda.synchronize(device)

        # warm-up graph
        for _ in range(2):
            graph.replay()
        
        # actual measurements
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_grouped_gemm.append(np.mean(results_this_batch_size))
        
    torch.cuda.empty_cache()

    # No plotting here; just return results
    return row_sizes, results_sequntial, results_sequntial_no_graph, results_grouped_gemm


@torch.inference_mode()
def benchmark_triton_fused_moe(hidden_size, intermediate_size, num_experts, label):

    # Use the same batch sizes as other benchmarks
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512, 1024])

    num_repeats = 5

    # Expert weights in the layout expected by the Triton demo helpers
    # BCs: (E, H, 2I)  -> transpose to (E, 2I, H) for GEMM-1
    # Ds:  (E, I, H)   -> transpose to (E, H, I)  for GEMM-2 (pre-transpose)
    BCs, Ds = alloc_expert_weights(hidden_size, intermediate_size, num_experts, device, dtype=DTYPE)
    B1 = BCs.transpose(1, 2).contiguous()                 # (E, 2I, H)
    w2 = Ds.transpose(1, 2).contiguous()                  # (E, H, I)

    results_triton_fused = []

    for n_rows in row_sizes:
        results_this_batch_size = []

        # Inputs: per-expert tokens with variable sizes, then flattened to (M, H)
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        As_list = [torch.randn(bs, hidden_size, device=device, dtype=DTYPE).contiguous() for bs in batch_sizes_i]
        A_flat = torch.cat(As_list, dim=0).contiguous()   # (M, H)

        # Top-1 routing mapping each token to its expert deterministically
        top_k = 1
        m = int(sum(batch_sizes_i))
        topk_ids = torch.empty((m, top_k), device=device, dtype=torch.int32)
        # Fill expert ids per contiguous segment
        start = 0
        for e, bs in enumerate(batch_sizes_i):
            end = start + int(bs)
            topk_ids[start:end, 0] = int(e)
            start = end
        topk_weights = torch.ones((m, top_k), device=device, dtype=DTYPE)

        # Align tokens into kernel-friendly blocks
        BLOCK_SIZE_M = 64
        sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
            topk_ids, BLOCK_SIZE_M, num_experts
        )
        EM = int(num_tokens_post_padded.item())
        if EM == 0:
            results_triton_fused.append(0.0)
            continue

        # Output buffers; for GEMM-2, use B as (E, N=k, K=d_ff) = w2
        c1 = torch.empty((EM, 2 * intermediate_size), device=device, dtype=DTYPE)
        inter_buf = torch.empty((EM, intermediate_size), device=device, dtype=DTYPE)
        c2 = torch.empty((EM, hidden_size), device=device, dtype=DTYPE)

        # Warm-up to trigger JIT compilation and stabilize runtime
        def run_once():
            _invoke_fused_moe_kernel(
                A_flat,
                B1,
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
            inter_buf.copy_(_silu_and_mul(c1))
            _invoke_fused_moe_kernel(
                inter_buf,
                w2,
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

        # Warm-up a few times
        for _ in range(2):
            run_once()

        # Capture CUDA graph for the two Triton GEMMs + activation
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            run_once()
        torch.cuda.synchronize(device)

        # Warm-up graph
        for _ in range(2):
            graph.replay()

        # Timed replays
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))
        results_triton_fused.append(np.mean(results_this_batch_size))

    torch.cuda.empty_cache()

    return row_sizes, results_triton_fused


@torch.inference_mode()
def benchmark_deep_gemm_moe(hidden_size, intermediate_size, num_experts, label):
    if not HAS_DEEP_GEMM:
        return np.array([]), []

    # Check CUDA capability
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        print("Skipping DeepGemm benchmark: requires Hopper (SM90) or later.")
        return np.array([]), []

    # Same batch sizes
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512, 1024])
    num_repeats = 5

    # Weights
    # BCs: (E, H, 2I). Need (E, 2I, H) for DeepGemm (N, K)
    # Ds: (E, I, H). Need (E, H, I) for DeepGemm (N, K)
    BCs_bf16, Ds_bf16 = alloc_expert_weights(hidden_size, intermediate_size, num_experts, device, dtype=torch.bfloat16)
    
    # Prepare weights in FP8
    # GEMM 1 Weights: (G, N, K) = (E, 2I, H)
    BCs_fp8 = BCs_bf16.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
    # GEMM 2 Weights: (G, N, K) = (E, H, I)
    Ds_fp8 = Ds_bf16.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)

    # Scaling factors helpers
    ceil_div = lambda x, y: (x + y - 1) // y

    # GEMM 1 scaling factors (B)
    # N=2*intermediate_size, K=hidden_size
    gemm1_N = 2 * intermediate_size
    gemm1_K = hidden_size
    sfb_1 = torch.ones(num_experts, ceil_div(gemm1_N, 128), ceil_div(gemm1_K, 128), device=device, dtype=torch.float32)

    # GEMM 2 scaling factors (B)
    # N=hidden_size, K=intermediate_size
    gemm2_N = hidden_size
    gemm2_K = intermediate_size
    sfb_2 = torch.ones(num_experts, ceil_div(gemm2_N, 128), ceil_div(gemm2_K, 128), device=device, dtype=torch.float32)

    results_deep_gemm = []

    for n_rows in row_sizes:
        
        # Inputs
        ratios_local = expert_batch_size_ratios if len(expert_batch_size_ratios) == num_experts else [1.0] * num_experts
        batch_sizes_i = [max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local]
        # Concatenate inputs from all experts
        As_list = [torch.randn(bs, hidden_size, device=device, dtype=torch.bfloat16) for bs in batch_sizes_i]
        A_flat_bf16 = torch.cat(As_list, dim=0).contiguous()
        M = A_flat_bf16.shape[0]

        # m_indices: map each row to its expert group
        m_indices = torch.empty(M, device=device, dtype=torch.int32)
        start = 0
        for e, bs in enumerate(batch_sizes_i):
            end = start + bs
            m_indices[start:end] = e
            start = end
        
        # GEMM 1 Inputs
        # Quantize inputs using deep_gemm utility (simulating incoming FP8 or on-the-fly cast)
        A_fp8, sfa_1 = dg.per_token_cast_to_fp8(A_flat_bf16, use_ue8m0=False)
        
        # Buffers
        up_buf = torch.empty(M, gemm1_N, device=device, dtype=torch.bfloat16)
        
        # For GEMM 2, we need scaling factors for the intermediate activation
        # It will depend on M
        down_buf = torch.empty(M, gemm2_N, device=device, dtype=torch.bfloat16)

        def run_once():
            # GEMM 1
            dg.m_grouped_fp8_gemm_nt_contiguous((A_fp8, sfa_1), (BCs_fp8, sfb_1), up_buf, m_indices)
            
            # Activation (BF16)
            up1 = up_buf[:, :intermediate_size]
            up3 = up_buf[:, intermediate_size:]
            # Simple GLU (elementwise mul) as per run_expert reference
            glu = up1 * up3
            
            # Convert to FP8 for GEMM 2 using deep_gemm utility
            glu_fp8, sfa_2 = dg.per_token_cast_to_fp8(glu, use_ue8m0=False)
            
            # GEMM 2
            dg.m_grouped_fp8_gemm_nt_contiguous((glu_fp8, sfa_2), (Ds_fp8, sfb_2), down_buf, m_indices)

        # Warmup
        for _ in range(2):
            run_once()
        
        # Timing
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        for _ in range(num_repeats):
            run_once()
        end_evt.record()
        torch.cuda.synchronize()
        
        results_deep_gemm.append(start_evt.elapsed_time(end_evt) / num_repeats)

    return row_sizes, results_deep_gemm


@torch.inference_mode()
def benchmark_single_expert_reference(hidden_size, intermediate_size, num_experts, label):
    # Same batch sizes as other benchmarks
    row_sizes = np.array([1,2,4,8,16,32,64,128,256, 512, 1024])
    num_repeats = 5

    # Single expert weights
    BC = torch.randn(hidden_size, intermediate_size * 2, device=device, dtype=DTYPE).contiguous()
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=DTYPE).contiguous()

    results_single_scaled = []

    for n_rows in row_sizes:
        results_this_batch_size = []

        A = torch.randn(int(n_rows), hidden_size, device=device, dtype=DTYPE).contiguous()
        up_buf = torch.empty((int(n_rows), intermediate_size * 2), device=device, dtype=DTYPE).contiguous()
        down_buf = torch.empty((int(n_rows), hidden_size), device=device, dtype=DTYPE).contiguous()

        def run_once():
            torch.matmul(A, BC, out=up_buf)
            up1 = up_buf[:, :intermediate_size]
            up3 = up_buf[:, intermediate_size:]
            up1 *= up3
            _ = torch.matmul(up1, D, out=down_buf)

        # warm-up
        for _ in range(2):
            run_once()

        # capture cuda graph
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            run_once()
        torch.cuda.synchronize(device)

        # warm-up graph
        for _ in range(2):
            graph.replay()

        # timed replays
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            results_this_batch_size.append(start.elapsed_time(end))

        # Scale single-expert runtime by number of experts to form a reference
        results_single_scaled.append(float(num_experts) * float(np.mean(results_this_batch_size)))

    torch.cuda.empty_cache()
    return row_sizes, results_single_scaled


# hidden_sizes_k = np.array([6, 4, 7, 2, 4])
# hidden_sizes = hidden_sizes_k * 1024
# intermediate_sizes_k = np.array([16, 12, 2, 6, ])
# intermediate_sizes = intermediate_sizes_k * 1024
# models = ["Mixtral 8x22B", "Mixtral 8x7B", "Deepseek V3", "Qwen3-30B", "Qwen3-235B"]
# labels = [f"{model}: hidden={h}k, intermediate={i}k" for model, h, i in zip(models, hidden_sizes_k, intermediate_sizes_k)]
# num_experts_list = [8, 8, 8, 8, 8]
# expert_batch_size_ratios = [1, 0.5, 0.1, 0.2, 0.8, 1.5, 1.9, 0.7] # this must equals to num_experts
hidden_sizes_k = np.array([2, 4])
hidden_sizes = hidden_sizes_k * 1024
intermediate_sizes = [768, 1536]
models = ["Qwen3-30B", "Qwen3-235B"]
labels = [f"{model}: hidden={h}k, intermediate={i}" for model, h, i in zip(models, hidden_sizes_k, intermediate_sizes)]
num_experts_list = [8, 8]
# expert_batch_size_ratios = [1, 0.5, 0.1, 0.2, 0.8, 1.5, 1.9, 0.7] # this must equals to num_experts
expert_batch_size_ratios = [1] * 8

# Grouped bar chart per model (single figure with subplots)
n_models = len(models)
n_cols = 2
n_rows = int(np.ceil(n_models / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
axes = np.array(axes).ravel()

flops_series_per_model = []
time_series_grouped = []
time_series_seq_no_graph = []
time_series_triton = []
time_series_deep_gemm = []
time_series_single_scaled = []
model_names = []
model_colors = []

# Prepare CSV accumulation for comparison data
csv_header = [
    "model",
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "per_expert_batch_size",
    "flops_gflops",
    "single_expert_xE_ms",
    "sequential_graph_ms",
    "sequential_no_graph_ms",
    "grouped_gemm_ms",
    "triton_fused_moe_ms",
    "deep_gemm_ms",
    "dtype",
]
csv_rows = []

for idx, (hidden_size, intermediate_size, num_experts, label, model) in enumerate(zip(hidden_sizes, intermediate_sizes, num_experts_list, labels, models)):
    row_sizes, seq_means, seq_no_graph_means, grp_means = benchmark_grouped_gemm(hidden_size, intermediate_size, num_experts, label)
    row_sizes_t, triton_means = benchmark_triton_fused_moe(hidden_size, intermediate_size, num_experts, label)
    row_sizes_s, single_scaled_means = benchmark_single_expert_reference(hidden_size, intermediate_size, num_experts, label)
    
    # Run DeepGemm benchmark if fp8 is requested (or if we decide to run it generally if available)
    # The function handles availability checks internally.
    # Only run if dtype is fp8 to ensure fair comparison context or because it's specifically for fp8.
    if _DTYPE_STR == "fp8":
        row_sizes_dg, deep_gemm_means = benchmark_deep_gemm_moe(hidden_size, intermediate_size, num_experts, label)
    else:
        row_sizes_dg, deep_gemm_means = row_sizes, []

    if len(deep_gemm_means) == 0:
        deep_gemm_means = [0.0] * len(row_sizes)

    assert np.array_equal(row_sizes, row_sizes_t), "Row sizes mismatch between benchmarks"
    assert np.array_equal(row_sizes, row_sizes_s), "Row sizes mismatch for single-expert reference"
    
    x = np.arange(len(row_sizes))
    # Adjust width for 6 bars
    width = 0.14
    ax = axes[idx]
    
    # Determine labels based on dtype
    dtype_suffix = ""
    if _DTYPE_STR == "fp8":
        dtype_suffix = " (BF16)" # standard kernels run in BF16 when fp8 mode is active
    
    ax.bar(x - 2.5*width, single_scaled_means, width=width, label=f"Single Expert xE{dtype_suffix}")
    ax.bar(x - 1.5*width, seq_means, width=width, label=f"Sequential (graph){dtype_suffix}")
    ax.bar(x - 0.5*width, seq_no_graph_means, width=width, label=f"Sequential (no graph){dtype_suffix}")
    ax.bar(x + 0.5*width, grp_means, width=width, label=f"Grouped GEMM{dtype_suffix}")
    ax.bar(x + 1.5*width, triton_means, width=width, label=f"Triton Fused MoE{dtype_suffix}")
    ax.bar(x + 2.5*width, deep_gemm_means, width=width, label="DeepGemm (FP8)")

    # Use sparse xticks for readability
    if len(row_sizes) > 12:
        tick_idx = np.linspace(0, len(row_sizes) - 1, num=12, dtype=int)
    else:
        tick_idx = np.arange(len(row_sizes))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([int(row_sizes[i]) for i in tick_idx])
    ax.set_title(f"{label}, experts={num_experts}")
    ax.set_xlabel("per-expert batch size")
    ax.set_ylabel("Avg Execution Time (ms)")
    ax.legend(fontsize=8)

    # Collect FLOPs vs time data (for all methods)
    flops_total = 6.0 * float(num_experts) * float(hidden_size) * float(intermediate_size) * row_sizes.astype(np.float64)
    flops_series_per_model.append(flops_total / 1e9)  # in GFLOPs
    time_series_grouped.append(np.array(grp_means, dtype=np.float64))
    time_series_seq_no_graph.append(np.array(seq_no_graph_means, dtype=np.float64))
    time_series_triton.append(np.array(triton_means, dtype=np.float64))
    time_series_single_scaled.append(np.array(single_scaled_means, dtype=np.float64))
    time_series_deep_gemm.append(np.array(deep_gemm_means, dtype=np.float64))
    model_names.append(model)
    model_colors.append(plt.get_cmap('tab10')(idx))

    # Accumulate CSV rows for this model
    for j, bs in enumerate(row_sizes):
        flops_gflops = (6.0 * float(num_experts) * float(hidden_size) * float(intermediate_size) * float(bs)) / 1e9
        csv_rows.append([
            str(model),
            int(hidden_size),
            int(intermediate_size),
            int(num_experts),
            int(bs),
            float(flops_gflops),
            float(single_scaled_means[j]),
            float(seq_means[j]),
            float(seq_no_graph_means[j]),
            float(grp_means[j]),
            float(triton_means[j]),
            float(deep_gemm_means[j]),
            str(_DTYPE_STR),
        ])

# Hide any unused subplots
for ax in axes[len(models):]:
    ax.set_visible(False)

plt.tight_layout()
# Suffix output filenames with dtype
_fname_suffix = f"_{_DTYPE_STR}"
_cmp_path = f"ggemm_comparison{_fname_suffix}.png"
plt.savefig(_cmp_path, dpi=300)
print(f"Saved combined plot to {_cmp_path}")

# Write CSV comparison data
_csv_path = f"ggemm_comparison_data{_fname_suffix}.csv"
with open(_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)
print(f"Saved comparison data CSV to {_csv_path}")

# Create FLOPs vs Time line plot (all methods on one figure)
fig2, ax2 = plt.subplots(figsize=(9, 6))
# Plot per model with consistent color; two methods: Grouped GEMM vs Sequential (no graph)
for flops_g, t_grp, t_sng, t_tri, t_single, t_dg, name, color in zip(
    flops_series_per_model,
    time_series_grouped,
    time_series_seq_no_graph,
    time_series_triton,
    time_series_single_scaled,
    time_series_deep_gemm,
    model_names,
    model_colors,
):
    # Determine labels based on dtype
    dtype_suffix = ""
    if _DTYPE_STR == "fp8":
        dtype_suffix = " (BF16)"

    ax2.plot(flops_g, t_grp, marker='o', linestyle='-', color=color, label=f"{name} - Grouped GEMM{dtype_suffix}")
    ax2.plot(flops_g, t_sng, marker='s', linestyle=':', color=color, alpha=0.9, label=f"{name} - Sequential (no graph){dtype_suffix}")
    ax2.plot(flops_g, t_tri, marker='^', linestyle='--', color=color, alpha=0.9, label=f"{name} - Triton Fused MoE{dtype_suffix}")
    ax2.plot(flops_g, t_single, marker='x', linestyle='-.', color=color, alpha=0.7, label=f"{name} - Single Expert xE{dtype_suffix}")
    if np.sum(t_dg) > 0:
        ax2.plot(flops_g, t_dg, marker='*', linestyle='-', color=color, alpha=0.8, label=f"{name} - DeepGemm (FP8)")
ax2.set_xlabel("Estimated FLOPs per batch (GFLOPs)")
ax2.set_ylabel("Avg Execution Time (ms)")
ax2.set_title("Time vs Estimated FLOPs (Grouped GEMM vs Separate Kernels)")
ax2.legend(ncol=2, fontsize=8)
ax2.grid(True, linestyle='--', alpha=0.3)
fig2.tight_layout()
_flops_path = f"ggemm_flops_vs_time{_fname_suffix}.png"
fig2.savefig(_flops_path, dpi=300)
print(f"Saved FLOPs vs time plot to {_flops_path}")

if _prof is not None:
    _prof.stop()
    _prof.export_chrome_trace(trace_path)
    print(f"Exported Perfetto-compatible trace to {trace_path}")
