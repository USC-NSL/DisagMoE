import torch
import ctypes

from torch import Tensor
from typing import List, Tuple, Dict

def tensor_as_buf(buf: int, shape: List[int], dtype = torch.float16) -> Tensor:
    # TODO(hogura|20241003): change c_int16 to a dynamic type
    data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_int16))
    return Tensor(
        data, shape, dtype
    )