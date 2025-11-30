import numpy as np
import torch

def get_m_indices(batch_sizes: torch.Tensor,
                      expert_ids: torch.Tensor) -> torch.Tensor:
    """
    batch_sizes: [num_experts]  (CPU or CUDA, int32/int64)
    expert_ids:  [num_experts]  (CPU or CUDA, int32/int64), arbitrary IDs
    Returns: m_indices on same device as expert_ids, dtype int32.
    """
    # Move to CPU and get plain ints
    counts = batch_sizes.to("cpu").to(torch.long).numpy()
    ids = expert_ids.to("cpu").to(torch.int32).numpy()

    # Fast C implementation of repeat_interleave
    m_indices_cpu = np.repeat(ids, counts).astype(np.int32, copy=False)

    # Single copy to target device
    return torch.from_numpy(m_indices_cpu).to(
        device=expert_ids.device, dtype=torch.int32, non_blocking=True
    )