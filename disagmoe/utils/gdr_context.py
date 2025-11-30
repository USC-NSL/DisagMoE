from disagmoe_c import GdrContext as GdrContextImpl

import torch

class GdrContext:
    
    def __init__(self, tensor: torch.Tensor):
        self.gdr_context = GdrContextImpl(tensor)
        self.tensor = tensor
        
    def copy_from_host(self, src: int, nbytes: int, dst_offset: int = 0) -> None:
        self.gdr_context.copy_from_host(src, nbytes, dst_offset)
        
    def copy_from_host_tensor(self, src: torch.Tensor) -> None:
        self.gdr_context.copy_from_host_tensor(src)
        
    def copy_to_host(self, dest: int, nbytes: int, src_offset: int = 0) -> None:
        self.gdr_context.copy_to_host(dest, nbytes, src_offset)
        
    def copy_to_host_tensor(self, dst: torch.Tensor) -> None:
        self.gdr_context.copy_to_host_tensor(dst)
        
    def fill(self, value: int, nbytes: int, dst_offset: int = 0) -> None:
        self.gdr_context.fill(value, nbytes, dst_offset)
        
    def copy_from_host_int32(self, src: list[int]) -> None:
        self.gdr_context.copy_from_host_int32(src)
        
    def copy_from_host_int64(self, src: list[int]) -> None:
        self.gdr_context.copy_from_host_int64(src)
        
    def copy_to_host_int32(self, nelems: int) -> list[int]:
        return self.gdr_context.copy_to_host_int32(nelems)
        
    def copy_to_host_int64(self, nelems: int) -> list[int]:
        return self.gdr_context.copy_to_host_int64(nelems)