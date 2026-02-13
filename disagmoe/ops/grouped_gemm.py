import torch
import disagmoe_c
from disagmoe.utils.logger import get_logger

_initialized: bool = False

def ensure_initialized():
    """Probe hardware once and select the optimal CUTLASS tile config."""
    global _initialized
    if not _initialized:
        device_id = torch.cuda.current_device()
        desc = disagmoe_c.init_grouped_gemm(device_id)
        get_logger().info(f"CUTLASS grouped GEMM initialized: {desc}")
        _initialized = True
