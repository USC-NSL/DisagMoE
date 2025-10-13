from grouped_gemm.backend import gmm, get_arguments, gmm_with_arguments
import torch
import numpy as np
import triton
import triton.language as tl
from typing import Optional

# triton grouped gemm copied from sglang
@triton.jit
def compute_m_range(
    pid,
    batch_size,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    BLOCK_SIZE_M: tl.constexpr,
):
    idx = 0
    for bs in range(batch_size):
        tiles = tl.load(m_num_tiles_indptr + bs)
        if pid >= tiles:
            idx = bs

    idx_start = tl.load(m_num_tiles_indptr + idx)

    m_range_start = tl.load(seg_indptr + idx) + (pid - idx_start) * BLOCK_SIZE_M
    m_range_end = min(tl.load(seg_indptr + idx + 1), m_range_start + BLOCK_SIZE_M)
    expert_id = tl.load(weight_indices + idx)
    return m_range_start, m_range_end, expert_id

@triton.jit
def grouped_gemm_triton_kernel(
    a,
    b,
    c,
    batch_size,
    N,
    K,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    use_fp8_w8a8,
    scale_a,
    scale_b,
    a_stride_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_stride_1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    c_dtype = c.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m_block = tl.load(m_num_tiles_indptr + batch_size)
    if pid_m >= total_m_block:
        return

    m_range_start, m_range_end, expert_id = compute_m_range(
        pid_m, batch_size, seg_indptr, weight_indices, m_num_tiles_indptr, BLOCK_SIZE_M
    )
    if m_range_end - m_range_start == 0:
        return

    n_range_start = pid_n * BLOCK_SIZE_N
    n_range_end = min(n_range_start + BLOCK_SIZE_N, N)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    offs_am = tl.where(offs_am < m_range_end - m_range_start, offs_am, 0)
    offs_bn = tl.where(offs_bn < n_range_end - n_range_start, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptr = a + (m_range_start + offs_am[:, None]) * a_stride_0 + offs_k[None, :]
    b_ptr = b + (
        (expert_id * b_stride_0)
        + (n_range_start + offs_bn[:, None]) * b_stride_1
        + offs_k[None, :]
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(
            a_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )
        b_tile = tl.load(
            b_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )
        accumulator = tl.dot(a_tile, b_tile.T, accumulator)
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K

    if use_fp8_w8a8:
        scale_a_value = tl.load(scale_a + expert_id)
        scale_b_value = tl.load(scale_b + expert_id)
        accumulator *= scale_a_value * scale_b_value
    c_tile = accumulator.to(c_dtype)

    offs_cm = m_range_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_range_start + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m_range_end) & (offs_cn[None, :] < n_range_end)
    tl.store(c_ptr, c_tile, mask=c_mask)


@triton.jit
def compute_m_num_tiles_indptr(
    m_num_tiles_indptr, seg_indptr, batch_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    for bs in range(batch_size):
        m = tl.load(seg_indptr + bs + 1) - tl.load(seg_indptr + bs)
        cur_num_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        pre_num_tiles = tl.load(m_num_tiles_indptr + bs)
        tl.store(m_num_tiles_indptr + bs + 1, pre_num_tiles + cur_num_tiles)


def grouped_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    batch_size: int,
    weight_column_major: bool,
    seg_indptr: Optional[torch.Tensor] = None,
    weight_indices: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
):
    assert weight_column_major == True  # TODO: more
    if use_fp8_w8a8:
        assert scale_a is not None and scale_b is not None

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
    }

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1,)](
        m_num_tiles_indptr, seg_indptr, batch_size, config["BLOCK_SIZE_M"]
    )

    grid = lambda META: (
        triton.cdiv(a.size(0), META["BLOCK_SIZE_M"]) + batch_size,
        triton.cdiv(b.size(1), META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_triton_kernel[grid](
        a,
        b,
        c,
        batch_size,
        b.size(1),
        b.size(2),
        seg_indptr,
        weight_indices,
        m_num_tiles_indptr,
        use_fp8_w8a8,
        scale_a,
        scale_b,
        a.stride(0),
        b.stride(0),
        b.stride(1),
        **config,
    )
    return c


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)

hidden_size = 4 * 1024
intermediate_size = 12 * 1024

BC2 = torch.rand(2, hidden_size, intermediate_size * 2, device=device)
D2 = torch.rand(2, intermediate_size, hidden_size, device=device)

def grouped_gemm2(x, batch_sizes):
    t1 = gmm(x, BC2, batch_sizes)
    t2 = t1[:, :intermediate_size] * t1[:, intermediate_size:]
    _ = gmm(t2, D2, batch_sizes)
    

BC = torch.rand(hidden_size, intermediate_size * 2, device=device)
D = torch.rand(intermediate_size, hidden_size, device=device)
    
def gemm(x):
    t1 = torch.matmul(x, BC)
    t2 = t1[:, :intermediate_size] * t1[:, intermediate_size:]
    _ = torch.matmul(t2, D)

def benchmark(func, name):
    results = []
    num_repeats = 5
    num_runs = 20
    
    for _ in range(2):
        func()
    
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph, stream=stream):
        func()
    torch.cuda.synchronize(device)
    for _ in range(2):
        graph.replay()
        
    for _ in range(num_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(num_runs):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        total_time = start.elapsed_time(end)
        avg_time = total_time / num_runs
        results.append(avg_time)
    print(f"{name} cost: {np.mean(results)}")
    
num_tokens = 200

x = torch.randn((num_tokens, hidden_size), device=device)

def run_gemm():
    gemm(x)
    
x2 = torch.randn((num_tokens * 2, hidden_size), device=device)
batch_sizes = torch.tensor([num_tokens, num_tokens], device=device)
def run_grouped_gemm():
    grouped_gemm2(x2, batch_sizes)

batch_sizes0 = torch.tensor([num_tokens, 0], device=device)
def run_grouped_gemm0():
    grouped_gemm2(x, batch_sizes0)
    
# BC_t = torch.rand(hidden_size, intermediate_size * 2, device=device)
# D_t = torch.rand(intermediate_size, hidden_size, device=device)
# def grouped_gemm_triton():
    
benchmark(run_gemm, "gemm")
benchmark(run_grouped_gemm, "grouped_gemm")
benchmark(run_grouped_gemm0, "grouped_gemm0")

max_batch_size = 512
cache_up = torch.empty((max_batch_size, intermediate_size * 2), device=device)
cache_down = torch.empty((max_batch_size, hidden_size), device=device)

cutlass_workspace_size, arguments_ptr = get_arguments(2, device)
cutlass_workspace = torch.empty(
    [cutlass_workspace_size], dtype=torch.uint8, device=device)

def _gmm(hiddens, weight, batch_sizes, **kwargs):
    return gmm_with_arguments(hiddens, weight, batch_sizes, cutlass_workspace, arguments_ptr, **kwargs)

gmm_with_cache = _gmm

def grouped_gemm_with_cache(bs, x, batch_sizes):
    up = gmm_with_cache(x, BC2, batch_sizes, c=cache_up)
    up = up[:bs, :intermediate_size] * up[:bs, intermediate_size:]
    down = gmm_with_cache(up, D2, batch_sizes, c=cache_down)
    
x2 = torch.randn((num_tokens * 2, hidden_size), device=device)
batch_sizes = torch.tensor([num_tokens, num_tokens], device=device)

def run_grouped_gemm_with_cache():
    grouped_gemm_with_cache(num_tokens * 2, x2, batch_sizes)
    

batch_sizes0 = torch.tensor([num_tokens, 0], device=device)
def run_grouped_gemm_with_cache0():
    grouped_gemm_with_cache(num_tokens, x, batch_sizes0)

benchmark(run_grouped_gemm_with_cache, name="run_grouped_gemm_with_cache")

benchmark(run_grouped_gemm_with_cache0, name="run_grouped_gemm_with_cache0")
    
