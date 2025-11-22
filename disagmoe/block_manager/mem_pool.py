# @copyright SGLang
# copied and adapted from sglang/srt/mem_cache/mem_pool.py and sglang/srt/mem_cache/allocator.py
import torch
import triton
import triton.language as tl
import numpy as np
from typing import List, Union, Optional, Tuple
from disagmoe.utils.utils import next_power_of_2

class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
    ):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )

        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values
        
    def write_loc(self, req_ids: torch.Tensor, seq_lens: torch.Tensor, values: torch.Tensor):
        self.req_to_token[req_ids, seq_lens] = values
        
    def get_last_loc(self, req_ids: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        return self.req_to_token[req_ids, seq_lens - 2]
    
    def get_latest_loc(self, req_ids: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        return self.req_to_token[req_ids, seq_lens - 1]

    def available_size(self):
        return len(self.free_slots)
    
    def get_block_table(self, req_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        return self.req_to_token[req_ids, : max_seq_len]
    
    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))

class KVCache:
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def get_cpu_copy(self, indices):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError()

class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        
        self.kv_cache_buffer = [
            torch.zeros(
                (2, self.size + 1, self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

    def _clear_buffers(self):
        del self.kv_cache_buffer
        
    def get_kv_buffer(self, layer_id: int):
        return self.kv_cache_buffer[layer_id]
    
    # def get_cpu_copy(self, indices):
    #     torch.cuda.synchronize()
    #     kv_cache_cpu = []
    #     chunk_size = self.cpu_offloading_chunk_size
    #     for layer_id in range(self.layer_num):
    #         kv_cache_cpu.append([])
    #         for i in range(0, len(indices), chunk_size):
    #             chunk_indices = indices[i : i + chunk_size]
    #             k_cpu = self.k_buffer[layer_id][chunk_indices].to(
    #                 "cpu", non_blocking=True
    #             )
    #             v_cpu = self.v_buffer[layer_id][chunk_indices].to(
    #                 "cpu", non_blocking=True
    #             )
    #             kv_cache_cpu[-1].append([k_cpu, v_cpu])
    #     torch.cuda.synchronize()
    #     return kv_cache_cpu

    # def load_cpu_copy(self, kv_cache_cpu, indices):
    #     torch.cuda.synchronize()
    #     chunk_size = self.cpu_offloading_chunk_size
    #     for layer_id in range(self.layer_num):
    #         for i in range(0, len(indices), chunk_size):
    #             chunk_indices = indices[i : i + chunk_size]
    #             k_cpu, v_cpu = (
    #                 kv_cache_cpu[layer_id][i // chunk_size][0],
    #                 kv_cache_cpu[layer_id][i // chunk_size][1],
    #             )
    #             assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
    #             k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
    #             v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
    #             self.k_buffer[layer_id][chunk_indices] = k_chunk
    #             self.v_buffer[layer_id][chunk_indices] = v_chunk
    #     torch.cuda.synchronize()

class BaseTokenToKVPoolAllocator:
    
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def clear(self):
        raise NotImplementedError()

    def alloc(self, need_size: int):
        raise NotImplementedError()

    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)

@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    ret_values,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Return value
    if pid == tl.num_programs(0) - 1:
        tl.store(ret_values, sum_num_new_pages)

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, need_sort)
        self.num_pages = size // page_size
        self.ret_values = torch.empty((), dtype=torch.int64, device=self.device)
        self.seen_max_num_extend_tokens_next_power_of_2 = 1
        self.clear()
        self.debug_mode = False

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.ret_values,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = self.ret_values.item()
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)