import torch
from disagmoe_c import rebind_1d_tensor, rebind_2d_tensor

GPU_PAGE_SIZE = 1 << 16

def get_cuda_aligned_tensor(numel: int, dtype: torch.dtype, alignment: int = GPU_PAGE_SIZE, device: str = "cuda"):
    """
    Allocate a CUDA tensor with a 64KB-aligned data pointer.
    Returns (aligned_tensor, base_tensor).
    
    Args:
        numel (int): Number of elements (not bytes).
        dtype (torch.dtype): Tensor dtype (e.g. torch.int32, torch.int64, torch.float32).
        alignment (int): Alignment in bytes (default 64KB).
    """
    # Element size in bytes
    elem_size = torch.tensor([], dtype=dtype).element_size()
    size_bytes = numel * elem_size

    # 1. Overallocate to ensure we have alignment slack
    buf = torch.empty(size_bytes + alignment, dtype=torch.uint8, device=device)

    # 2. Compute aligned start address
    base_addr = buf.data_ptr()
    aligned_addr = (base_addr + alignment - 1) & ~(alignment - 1)
    offset = aligned_addr - base_addr

    # 3. Slice the overallocated buffer to create an aligned view
    aligned_buf = buf[offset:offset + size_bytes]

    # 4. Reinterpret as the requested dtype
    aligned_tensor = aligned_buf.view(dtype)
    
    # 5. Sanity check
    assert aligned_tensor.data_ptr() % alignment == 0, "Alignment failed!"

    return aligned_tensor

def make_tensor_view(dtype: torch.dtype, device: str = "cuda") -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device=device)

def bind_tensor_view_1d(output: torch.Tensor, base: torch.Tensor, length: int) -> torch.Tensor:
    rebind_1d_tensor(output, base, 0, length)
    
def bind_tensor_view_2d(
    output: torch.Tensor, 
    base: torch.Tensor, 
    offset: int,
    rows: int,
    cols: int,
    row_stride: int,
    col_stride: int = 1
) -> torch.Tensor:
    rebind_2d_tensor(output, base, 0, rows, cols, row_stride, col_stride)