import torch
import time
import enum
import os
import asyncio

from disagmoe.executor.executor import Executor, ExpertsExecutor, AttnExecutor
from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.frontend.adapter import Scheduler, MuDispatcher, Sampler, Tokenizer, BlockManager
from disagmoe.frontend.datatypes import (Metadata, ChannelInfo, TensorBatch,
                                         AttentionBatchMetadata, SloStat, TraceContext)
from disagmoe.ops.memory import get_mappings_from_exp_ids, permute_tokens_cuda as permute_tokens
from disagmoe.utils.logger import get_logger
from disagmoe.utils.utils import (get_ip, get_nccl_url_from_uid, 
                                  make_seqlens_cuda_tensor, make_seqlens_list, get_graph_batch_size, StepInfo, 
                                  nvtx_range, range_push, range_pop)
from disagmoe.utils.constants import *
from disagmoe.utils.placement import ParallelConfig
from disagmoe.models.utils import (pack_flash_attn_meta, unpack_flash_attn_meta, 
                                   make_prefill_meta, make_dummy_meta, CudaGraphContext)
from disagmoe.models.distributed import set_tensor_model_parallel_config, set_tensor_model_parallel_channel, group_sync
from disagmoe.env import ENV_VARS

from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from typing import Optional, List, Dict, Union, Callable, Tuple
from threading import Thread

from torch import Tensor

import torch.distributed as dist

from disagmoe_c import (init_engine, start_engine, init_sampler, init_tokenizer, set_hosts,
                        TensorBatch as TensorBatch_C,
                        BlockManager as BlockManager_C,
                        recorder_create as disagmoe_recorder_create,
                        recorder_output as disagmoe_recorder_output)

class EngineType(enum.Enum):
    ATTENTION = enum.auto()
    EXPERT = enum.auto()
    TOKENIZER = enum.auto()
    SAMPLER = enum.auto()

class Engine:

    def __init__(self, 
                 scheduler: Optional[Scheduler] = None, 
                 executor: Optional[Executor] = None, 
                 dispatcher: Optional[MuDispatcher] = None, 
                 device_id: Optional[int] = None):
        self.scheduler: Scheduler = scheduler
        self.attn_scheduler: Scheduler = scheduler
        self.executor: Executor = executor
        self.dispatcher: MuDispatcher = dispatcher
        self.device_id = device_id
        self.end_flag = False
        self.engine_type: EngineType = None
        self.model_config: ModelConfig = None
        self.cache_config: CacheConfig = None
        
        if device_id is not None:
            self._logger = get_logger(f"engine{device_id}")
            
        self.loop_thread = Thread(target=self.loop)
        
        self._process_batch: Callable
        
        self.block_mgr: BlockManager = None
        
        self.decode_seq_lens = {}
        self.profiler = None
        self.inner_exp_rank = []
        self.device_group_ids = []
        self.handles = []
        self.rank_in_group = 0 # EP rank in expert worker, TP rank in attention worker
        
        # for stats usage
        self._batch_sizes = []
        self._step_stats = []

    @property
    def is_attn(self):
        return self.engine_type == EngineType.ATTENTION
    
    @property
    def is_attn_driver(self):
        return self.is_attn and (self._inter_group_tp_enabled or self.rank_in_group == 0)
    
    @property
    def is_attn_worker(self):
        return self._intra_group_tp_enabled and self.rank_in_group > 0
    
    @property
    def _tp_enabled(self):
        return self.is_attn and self.model_config.tp_size > 1
    
    @property
    def _inter_group_tp_enabled(self):
        return self._tp_enabled and self.model_config.tp_enable_inter_group
    
    @property
    def _intra_group_tp_enabled(self):
        return self._tp_enabled and (not self.model_config.tp_enable_inter_group)
        
    def init_core(
            self,
            layer_ids: List[int],
            # P2P Channels
            in_device_ids: List[int],
            out_device_ids: List[int],
            out_channel_infos: List[ChannelInfo],
            # Group Channels
            in_nccl_ids: Dict[int, int],
            out_device_group_ids: Dict[int, List[int]],
            out_nccl_ids: Dict[int, int],
            device_group_ids: List[int] = None,
            group_nccl_ids: Tuple[str, str, str] = ("", "", ""),
            expert_ranks: List[Tuple[int, int, int]] = [],
            local_attn_dp_rank: int = 0,
        ):
        """
        NOTE(hogura|20241003): When using ray, all the device_id called to CUDA should become 0
        """
        disagmoe_recorder_create()
        
        self.model_config.layer_ids = layer_ids
        self.device_group_ids = device_group_ids
        
        if self.engine_type == EngineType.ATTENTION:
            self.executor = AttnExecutor.build(self.model_config, self.cache_config)
            
            if self.cache_config.num_gpu_blocks is None:
                self.cache_config.num_gpu_blocks = self.determine_kv_cache_blocks() // len(self.model_config.layer_ids)
                self._logger.info(f"kv cache num_gpu_blocks: {self.cache_config.num_gpu_blocks}")
            self.executor.initialize_cache(self.cache_config.num_gpu_blocks)
            self.cache_config.num_gpu_blocks -= self.cache_config.num_reserved_blocks
            self._process_batch = self.process_batch_attn
            self.block_mgr = BlockManager_C(
                self.cache_config.block_size, 
                self.cache_config.num_gpu_blocks, 
                self.cache_config.num_reserved_blocks
            )
            if self._intra_group_tp_enabled:
                self._create_broadcast_buffers()
        elif self.engine_type == EngineType.EXPERT:
            self.executor = ExpertsExecutor(self.model_config)
            self._process_batch = self.process_batch_expert
            # prepare inner exp rank, [n_exp_per_rank * rank, (rank + 1) * n_exp_per_rank) -> [0, n_exp_per_rank)
            self.inner_exp_rank = [0 for _ in range(self.model_config.num_experts_per_rank)]
            for i in range(self.model_config.num_experts_per_rank):
                self.inner_exp_rank[i] = self.model_config.num_experts_per_rank * self.rank_in_group + i
                
        if not self.model_config.tp_enable_inter_group:
            device_group_ids = None
            out_device_group_ids = {}
            
        self._logger.info(f"launching core: {layer_ids, in_device_ids, \
                          out_device_ids, out_channel_infos, \
                          out_device_group_ids, \
                          device_group_ids, expert_ranks, local_attn_dp_rank}")
        if device_group_ids is None:
            device_group_ids = []
        self.scheduler, self.attn_scheduler, self.dispatcher = init_engine(
            self.device_id,
            self.is_attn,
            layer_ids,
            # P2P Channels
            in_device_ids,
            out_device_ids,
            [info.to_c() for info in out_channel_infos],
            # Parallel config
            ParallelConfig.from_c(
                self.model_config.tp_size if self.model_config.tp_enable_inter_group else 1, # control the init of attn_scheduler
                self.model_config.ep_size,
                self.model_config.dp_size,
                self.model_config.num_experts_per_rank,
                expert_ranks,
            ),
            # Group Channels
            in_nccl_ids,
            out_device_group_ids,
            out_nccl_ids,
            device_group_ids,
            group_nccl_ids,
            local_attn_dp_rank,
        )
        if self.model_config.tp_enable_inter_group:
            set_tensor_model_parallel_channel(self.attn_scheduler.get_channel() if self.attn_scheduler is not None else None)
        else:
            if self.is_attn:
                dist.init_process_group(backend="nccl", 
                                        world_size=len(self.device_group_ids), 
                                        rank=self.rank_in_group,
                                        init_method=f"tcp://{get_nccl_url_from_uid(group_nccl_ids[0])}")
        
        if self.is_attn:
            self.attn_scheduler.set_max_batch_size(self.max_batch_size)
        else:
            self.scheduler.set_max_batch_size(self.max_batch_size)
        
        self._warmup()
        if self.model_config.enable_cuda_graph_attn or self.model_config.enable_cuda_graph_expert:
            self._create_cuda_graph_contexts()
            self._cuda_graph_capture()
        
        self._logger.info("core launched")
        
    def _switch_scheduler(self):
        if self.scheduler is None:
            self.scheduler = self.attn_scheduler
    
    def start(self):
        if self.is_attn_worker:
            self.loop_thread = Thread(target=self.attn_worker_loop)
        else:
            start_engine(self.scheduler, self.attn_scheduler, self.dispatcher)
            self._switch_scheduler()
            
        self.loop_thread.start()

    def set_device_id(self, device_id: int):
        self.device_id = device_id
        self._logger = get_logger(f"engine{device_id}")

    def set_hosts(self, device_2_host: Dict[int, str]):
        device_2_host[self.device_id] = "0.0.0.0"
        set_hosts(os.getpid(), device_2_host)

    def setup_engine(self, 
                     engine_type: EngineType,
                     model_config: ModelConfig,
                     cache_config: CacheConfig = None,
                     rank: int = 0):
        self.rank_in_group = rank
        torch.set_default_dtype(torch.bfloat16)
        if engine_type in [EngineType.ATTENTION, EngineType.EXPERT]:
            torch.set_default_device("cuda:0")
            stream = torch.cuda.Stream(priority=-1)
            torch.cuda.set_stream(stream)
            self._logger.info(f"set stream {stream}")
            self.stream = stream
            set_tensor_model_parallel_config(model_config)
            free_memory, total_memory = torch.cuda.mem_get_info()
            self._logger.info(f"CUDA free memory: {free_memory / (1024 ** 3):.2f} GB, "\
                              f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
            self.init_gpu_memory = free_memory
            
        self.engine_type = engine_type
        self.model_config = model_config
        self.cache_config = cache_config
        self.max_batch_size = model_config.max_batch_size_attn \
            if engine_type == EngineType.ATTENTION \
            else model_config.max_batch_size_expert
        
        self._logger.info(f"engine setup. {self.engine_type, model_config}")
    
    def determine_kv_cache_blocks(self) -> int:
        assert isinstance(self.executor, AttnExecutor)
        torch.cuda.empty_cache()
        
        self.executor.profile_execute(self.max_batch_size)      
        torch.cuda.synchronize()  
        
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = self.init_gpu_memory - free_gpu_memory
        cache_block_size = self.model_config.hidden_size * self.cache_config.block_size * 2 # fp16 or bf16
        
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        
        return num_gpu_blocks

    def _create_cuda_graph_contexts(self):
        # assert self.model_config.enable_cuda_graph_attn or self.model_config.enable_cuda_graph_expert
        batch_size = self.max_batch_size
        self.graphs: List[List[torch.cuda.CUDAGraph]] = []
        self.static_input = torch.zeros((batch_size, self.model_config.hidden_size), device="cuda")
        self.static_output = torch.zeros((batch_size, self.model_config.hidden_size), device="cuda")
        if self.is_attn:
            self.static_expert_ids = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")
            self.static_positions = torch.zeros(batch_size, dtype=torch.long, device="cuda")
            self.static_block_table = torch.zeros(
                (batch_size, self.cache_config.num_gpu_blocks), 
                dtype=torch.int32, device="cuda")
            self.static_slot_mapping = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")
            self.static_batch_infos = torch.zeros(
                # (num_seqs + num_seqs + (meta_py.num_prefill_seqs + 1) + (num_seqs + 1)), 
                (batch_size + batch_size + (batch_size + 1) + (batch_size + 1)), 
                dtype=torch.int32, device="cuda")
        else:
            self.static_batch_sizes = torch.zeros((self.model_config.num_experts_per_rank, ), dtype=torch.long, 
                                                  device="cuda" if ENV_VARS["GROUPED_GEMM_CUTLASS"] else "cpu")
            self.static_mappings_gpu = torch.zeros((batch_size, ), dtype=torch.int32, device="cuda")
            
        self.graph_batch_sizes = list(range(max(self.model_config.graph_stride, self.model_config.ep_size),
                                            batch_size + 1,
                                            self.model_config.graph_stride))
        if self.is_attn:
            self.graph_batch_sizes = [1] + self.graph_batch_sizes
        for i in range(self.model_config.num_layers):
            self.graphs.append([
                torch.cuda.CUDAGraph() for _ in self.graph_batch_sizes
            ])
            
    def _warmup(self):
        if self.is_attn:
            self._warmup_attn()
        else:
            self._warmup_experts()
            
    def _warmup_attn(self):
        # a hacking to cuda graph, since `_pack_flash_attn_metadata` may have different behaviors.
        _enable_cuda_graph_attn = self.model_config.enable_cuda_graph_attn
        self.model_config.enable_cuda_graph_attn = False
        
        input = torch.zeros((self.max_batch_size, self.model_config.hidden_size), device="cuda")
        positions = torch.zeros(self.max_batch_size, dtype=torch.long, device="cuda")
        meta_py = make_dummy_meta(self.max_batch_size, 0)
        meta = self._pack_flash_attn_metadata(meta_py.to_c(), meta_py, [], mocking=True)
        for layer_id in self.model_config.layer_ids:
            for _ in range(2):
                _, _ = self.executor.execute(layer_id, positions, input, meta)
        
        # restore the original setting
        self.model_config.enable_cuda_graph_attn = _enable_cuda_graph_attn

    def _warmup_experts(self):
        input = torch.zeros((self.max_batch_size, self.model_config.hidden_size), device="cuda")
        batch_sizes = torch.tensor([self.max_batch_size // len(self.inner_exp_rank)] * len(self.inner_exp_rank),
            dtype=torch.int64,
            # NOTE(hogura|20241014): cuBLAS grouped_gemm requires batch_sizes to be on cpu
            device="cuda" if ENV_VARS["GROUPED_GEMM_CUTLASS"] else "cpu")
        for layer_id in self.model_config.layer_ids:
            for _ in range(2):
                _ = self.executor.execute(layer_id, input, batch_sizes)
            
    def _cuda_graph_capture(self):
        if self.is_attn:
            self._cuda_graph_capture_attn()
        else:
            self._cuda_graph_capture_experts()

    def _cuda_graph_capture_attn(self):
        if not self.model_config.enable_cuda_graph_attn:
            self._logger.warning("Attention CUDA Graph is not enabled.")
            return
        
        meta_py = make_dummy_meta(self.max_batch_size, 0)
        meta = self._pack_flash_attn_metadata(meta_py.to_c(), meta_py, [], mocking=True)
        if self.model_config.tp_size > 1:
            group_sync()
            
        meta.use_cuda_graph = True
        for layer_id in self.model_config.layer_ids:
            for graph, graph_batch_size in zip(self.graphs[layer_id], self.graph_batch_sizes):
                meta_py = make_dummy_meta(graph_batch_size, 0)
                meta = self._pack_flash_attn_metadata(meta_py.to_c(), meta_py, [], mocking=True)
                self._logger.info(f"Attention CUDA Graph capturing layer {layer_id, graph_batch_size}")
                with torch.cuda.graph(graph):
                    self.static_output, self.static_expert_ids = self.executor.execute(
                        layer_id, self.static_positions, self.static_input[: graph_batch_size], meta)
                torch.cuda.synchronize()
                
                if self.model_config.tp_size > 1:
                    group_sync()
                    
                # warmup for the actual execution
                for _ in range(2):
                    graph.replay()
                    
                torch.cuda.synchronize()
    
    def _cuda_graph_capture_experts(self):
        if not self.model_config.enable_cuda_graph_expert:
            self._logger.warning("Expert CUDA Graph is not enabled.")
            return
        if ENV_VARS["GROUPED_GEMM_CUTLASS"] != "1":
            self._logger.critical("Expert CUDA Graph requires CUTLASS grouped_gemm. " \
                                  "Unexpected behaviors may exist when using cuBLAS Grouped GEMM.")
        assert self.model_config.enable_grouped_gemm or self.model_config.num_experts_per_rank == 1, \
            "Serial GEMM could not be captured by CUDA Graph when num_experts_per_rank > 1"
        
        for layer_id in self.model_config.layer_ids:
            for graph, graph_batch_size in zip(self.graphs[layer_id], self.graph_batch_sizes):
                self.static_batch_sizes.fill_(graph_batch_size // self.model_config.num_experts_per_rank)
                self.static_batch_sizes[-1] += graph_batch_size % self.model_config.num_experts_per_rank
                self._logger.info(f"Expert CUDA Graph capturing layer {layer_id, graph_batch_size}")
                with torch.cuda.graph(graph):
                    self.static_output = self.executor.execute(layer_id, self.static_input[: graph_batch_size], self.static_batch_sizes)
                torch.cuda.synchronize()
                
                for _ in range(2):
                    # warmup for the actual execution
                    graph.replay()
                    
                torch.cuda.synchronize()

    def _create_broadcast_buffers(self):
        self.buffer_meta = torch.empty((BROADCAST_BUFFER_SIZE), dtype=torch.int32, device="cuda")
        self.buffer_tensor = torch.empty((self.max_batch_size, self.model_config.hidden_size), device="cuda")
        
        # [decode_seq_lens, query_start_loc, seq_start_loc, context_lens, slot_mapping, block_table]
        shape = (self.max_batch_size + self.max_batch_size * MAX_SEQ_LEN // self.cache_config.block_size, )
        self.buffer_attn_meta = torch.empty(shape, dtype=torch.int32, device="cuda")
        
        self.buffer_meta.fill_(0)
        self.buffer_tensor.fill_(0)
        self.buffer_attn_meta.fill_(0)
        
    def _wait_async_handles(self):
        for h in self.handles:
            h.wait()
        self.handles = []
    
    def _add_async_handle(self, handle):
        self.handles.append(handle)
        
    @nvtx_range("engine._update_block_table")
    def _update_block_table(self, meta_c: AttentionBatchMetadata, meta_py: AttentionBatchMetadata) -> List[int]:
        prefill_seq_ids = meta_py.seq_ids[ : meta_py.num_prefill_seqs]
        decode_seq_ids = meta_py.seq_ids[meta_py.num_prefill_seqs : ]
            
        #  if the first layer in this attention worker, update block table and decode_seq_lens
        decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]
        
        if meta_py.layer_id == self.model_config.layer_ids[0]:
            # only update block table and decode_seq_lens in the first layer
            self.block_mgr.update_block_table(meta_c, decode_seq_lens)
            
            for i, seq_id in enumerate(prefill_seq_ids):
                self.decode_seq_lens[seq_id] = meta_py.prefill_seq_len[i]
            
            for i, seq_id in enumerate(decode_seq_ids):
                decode_seq_lens[i] += 1
                self.decode_seq_lens[seq_id] += 1
            
        return decode_seq_lens
    
    @nvtx_range("engine.pack_flash_attn_metadata")
    def _pack_flash_attn_metadata(
            self,
            meta_c: AttentionBatchMetadata,
            meta_py: AttentionBatchMetadata,
            decode_seq_lens: List[int],
            mocking: bool = False,
        ) -> FlashAttentionMetadata:
        
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens
        
        # 1. prepare block table
        if not mocking:
            block_table_1d = self.block_mgr.prepare_block_table(meta_c, decode_seq_lens)
        else:
            # mocking=True when _warmup_attn
            block_table_1d = torch.zeros(
                (num_tokens + num_seqs * MAX_SEQ_LEN // self.cache_config.block_size, ), 
                dtype=torch.int32, device="cuda")
        
        if self.model_config.enable_cuda_graph_attn:
            self.static_slot_mapping[0: num_seqs].copy_(
                block_table_1d[-num_seqs : ].to(torch.int64))
            _max_num_blocks = (block_table_1d.numel() - num_seqs) // num_tokens
            self.static_block_table[0: num_tokens, 0: _max_num_blocks].copy_(
                block_table_1d[ : -num_seqs].view(num_tokens, -1)
            )
            slot_mapping_cuda = self.static_slot_mapping
            block_table_cuda = self.static_block_table
        else:
            slot_mapping_cuda = block_table_1d[-num_seqs : ].to(torch.int64)
            block_table_cuda = block_table_1d[ : -num_tokens].view(num_tokens, -1)
        
        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, query_start_loc, seq_start_loc) in the same tensor
        batch_infos = [0] * (num_seqs + num_seqs + (meta_py.num_prefill_seqs + 1) + (num_seqs + 1)) 
        
        # make seq_lens
        batch_infos[ : meta_py.num_prefill_seqs] = meta_py.prefill_seq_len
        batch_infos[meta_py.num_prefill_seqs : num_seqs] = decode_seq_lens
        seq_lens = batch_infos[ : num_seqs]
        
        # make context_lens
        for i in range(meta_py.num_prefill_seqs):
            batch_infos[num_seqs + i] = meta_py.prefill_seq_len[i] - meta_py.prefill_query_len[i]
        for i in range(meta_py.num_decode_tokens):
            batch_infos[num_seqs + meta_py.num_prefill_seqs + i] = decode_seq_lens[i] - 1
            
        # make query_start_loc
        make_seqlens_list(meta_py.prefill_query_len, dst=batch_infos[num_seqs + num_seqs : num_seqs + num_seqs + meta_py.num_prefill_seqs + 1])

        # make seq_start_loc
        make_seqlens_list(seq_lens, dst=batch_infos[num_seqs + num_seqs + meta_py.num_prefill_seqs + 1 : ])

        batch_infos_cuda = torch.tensor(batch_infos, dtype=torch.int32, device="cuda")
        if self.model_config.enable_cuda_graph_attn:
            self.static_batch_infos[ : len(batch_infos)].copy_(batch_infos_cuda)
            batch_infos_cuda = self.static_batch_infos
        
        seq_lens_cuda = batch_infos_cuda[ : num_seqs]
        context_lens_tensor = batch_infos_cuda[num_seqs : num_seqs + num_seqs]
        query_start_loc = batch_infos_cuda[num_seqs + num_seqs : num_seqs + num_seqs + meta_py.num_prefill_seqs + 1]
        seq_start_loc = batch_infos_cuda[num_seqs + num_seqs + meta_py.num_prefill_seqs + 1 : ]

        max_query_len = max(meta_py.prefill_query_len) if len(meta_py.prefill_query_len) > 0 else 0
        max_prefill_seq_len = max(meta_py.prefill_seq_len) if len(meta_py.prefill_seq_len) > 0 else 0
        max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
        
        max_num_blocks = (max(seq_lens) - 1) // self.cache_config.block_size + 1
        assert mocking or self.model_config.enable_cuda_graph_attn or \
               max_num_blocks == block_table_cuda.shape[-1], "block table wrong"
        
        return FlashAttentionMetadata(
            meta_py.num_prefill_seqs,
            meta_py.num_prefill_tokens,
            meta_py.num_decode_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_table_cuda,
            use_cuda_graph=self.model_config.enable_cuda_graph_attn,
        )
    
    @nvtx_range("engine.attn_driver_preprocess")
    def _attn_driver_preprocess(self, 
                                meta_c: AttentionBatchMetadata, 
                                meta_py: AttentionBatchMetadata, 
                                input_tensor: Tensor) -> FlashAttentionMetadata:
        decode_seq_lens = self._update_block_table(meta_c, meta_py)
        
        if self._intra_group_tp_enabled:
            # 1. broadcast necessary metadata
            bc_meta = [
                meta_py.layer_id, # 0
                meta_py.num_prefill_seqs, # 1
                meta_py.num_prefill_tokens, # 2
                meta_py.num_decode_tokens, # 3
                *meta_py.prefill_query_len, # 4
                *meta_py.prefill_seq_len, # 4 + num_prefill_seqs
                *decode_seq_lens, # 4 + num_prefill_seqs + num_prefill_seqs
            ]
            
            self.buffer_meta[ : len(bc_meta)].copy_(torch.tensor(bc_meta, dtype=torch.int32, device="cpu"))
            dist.broadcast(self.buffer_meta, 0)
            
            # 2. broadcast input tensor asynchronously
            self._add_async_handle(dist.broadcast(input_tensor, 0, async_op=True))
            
        attn_meta = self._pack_flash_attn_metadata(meta_c, meta_py, decode_seq_lens)

        if self._intra_group_tp_enabled:
            self._wait_async_handles()
            
            # 3. broadcast attn_meta
            # [slot_mapping, block_table]
            max_num_blocks = attn_meta.block_tables.shape[-1]
            num_tokens = meta_py.num_prefill_tokens + meta_py.num_decode_tokens
            num_elems = num_tokens + max_num_blocks * num_tokens
            
            bc_attn_meta = self.buffer_attn_meta[ : num_elems]
            
            if not attn_meta.use_cuda_graph:
                bc_attn_meta[ : num_tokens].copy_(attn_meta.slot_mapping.to(torch.int32))
                bc_attn_meta[num_tokens : ].copy_(attn_meta.block_tables.view(-1))
            else:
                bc_attn_meta[ : num_tokens].copy_(
                    attn_meta.slot_mapping[ : num_tokens].to(torch.int32))
                self._logger.info(f"block_table shape: {attn_meta.block_tables.shape, num_tokens, max_num_blocks, bc_attn_meta.shape}")
                bc_attn_meta[num_tokens : ].copy_(
                    attn_meta.block_tables[ : num_tokens, : max_num_blocks].view(-1))
            
            dist.broadcast(bc_attn_meta, 0)
        
        return attn_meta
    
    @nvtx_range("engine.attn_worker_preprocess")
    def _attn_worker_preprocess(self) -> Tuple[int, Tensor, FlashAttentionMetadata]:
        dist.broadcast(self.buffer_meta, 0)
        meta = self.buffer_meta.tolist()
        layer_id = meta[0]
        if layer_id == -1:
            # terminated
            return -1, None, None
        num_prefill_seqs = meta[1]
        num_prefill_tokens = meta[2]
        num_decode_tokens = meta[3]
        batch_size = get_graph_batch_size(num_prefill_tokens + num_decode_tokens)[1] if self.model_config.enable_cuda_graph_attn \
            else num_prefill_tokens + num_decode_tokens
        
        num_tokens = num_prefill_tokens + num_decode_tokens
        num_seqs = num_prefill_seqs + num_decode_tokens
        
        input_tensor = self.buffer_tensor[ : num_prefill_tokens + num_decode_tokens]
        self._add_async_handle(dist.broadcast(input_tensor, 0, async_op=True))
        
        prefill_query_lens = meta[4 : 4 + num_prefill_seqs]
        if not self.model_config.enable_cuda_graph_attn:
            seq_lens = meta[4 + num_prefill_seqs : 4 + num_prefill_seqs + num_seqs]
            seq_lens_cuda = self.buffer_meta[4 + num_prefill_seqs : 4 + num_prefill_seqs + num_seqs]
        else:
            # extend seq_lens to batch_size
            seq_lens = meta[4 + num_prefill_seqs : 4 + num_prefill_seqs + num_seqs]
            for _ in range(batch_size - num_seqs):
                seq_lens.append(0)
            seq_lens_cuda = self.buffer_meta[4 + num_prefill_seqs : 4 + num_prefill_seqs + batch_size]
        prefill_seq_lens = seq_lens[ : num_prefill_seqs]
        decode_seq_lens = seq_lens[num_prefill_seqs : ]
        
        max_num_blocks = (max(seq_lens) - 1) // self.cache_config.block_size + 1
        # [slot_mapping, block_table]
        num_elems = num_tokens + max_num_blocks * num_tokens
        
        query_start_loc = make_seqlens_cuda_tensor(prefill_query_lens)
        seq_start_loc = make_seqlens_cuda_tensor(seq_lens)
        context_lens = [seq_len - que_len for seq_len, que_len in zip(prefill_seq_lens, prefill_query_lens)] + [seq_len - 1 for seq_len in decode_seq_lens]
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
        
        max_query_len = max(prefill_query_lens) if len(prefill_query_lens) > 0 else 0
        max_prefill_seq_len = max(prefill_seq_lens) if len(prefill_seq_lens) > 0 else 0
        max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
        
        self._wait_async_handles()
        
        bc_attn_meta = self.buffer_attn_meta[ : num_elems]
        dist.broadcast(bc_attn_meta, 0)
        
        if not self.model_config.enable_cuda_graph_attn:
            slot_mapping_cuda = bc_attn_meta[ : num_tokens].to(torch.int64)
            block_table_cuda = bc_attn_meta[num_tokens : ].view(num_tokens, -1)
        else:
            self.static_slot_mapping[ : num_tokens].copy_(bc_attn_meta[ : num_tokens].to(torch.int64))
            self.static_block_table[ : num_tokens, 0: max_num_blocks].copy_(bc_attn_meta[num_tokens : ].view(num_tokens, -1))
            slot_mapping_cuda = self.static_slot_mapping
            block_table_cuda = self.static_block_table
        
        return layer_id, input_tensor, FlashAttentionMetadata(
            num_prefill_seqs,
            num_prefill_tokens,
            num_decode_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_table_cuda,
            use_cuda_graph=self.model_config.enable_cuda_graph_attn,
        )

    @nvtx_range("engine.process_batch_attn")
    def process_batch_attn(self, 
                           meta_c: AttentionBatchMetadata, 
                           input_tensor: Tensor,
                           mocking: bool = False) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, AttnExecutor)
        
        # self._logger.debug(f"process batch AttentionBatchMetadata: {meta_c}")
        
        # TODO(hogura|20241014): fill the real positions
        positions = torch.zeros(input_tensor.shape[0], dtype=torch.long, device="cuda")
        # self._logger.info(f"process batch attn {meta_c.seq_ids}")
        
        if mocking:
            # if mocking is enabled, the meta_c is a python AttentionBatchMetadata class
            meta_py = meta_c
            meta_c = meta_c.to_c()
        else:
            meta_py = AttentionBatchMetadata.from_c(meta_c)
            assert len(meta_py.seq_ids) > 0, "Scheduled batch is empty"
        
        attn_meta = self._attn_driver_preprocess(meta_c, meta_py, input_tensor)
        
        # TODO(hogura|20241015): only top-1 expert currently
        # self._logger.info(f"executing attn {meta_c.seq_ids}")
        if not self.model_config.enable_cuda_graph_attn:
            hiddens, expert_ids = self.executor.execute(meta_c.layer_id, positions, input_tensor, attn_meta)
        else:
            num_tokens = input_tensor.shape[0]
            graph_id, batch_size = get_graph_batch_size(num_tokens, self.graph_batch_sizes)
            self.static_input[:num_tokens].copy_(input_tensor)
            range_push("engine.graph_replay")
            self.graphs[meta_c.layer_id][graph_id].replay()
            hiddens = self.static_output[:num_tokens]
            expert_ids = self.static_expert_ids[:num_tokens]
            torch.cuda.synchronize()
            range_pop()
        
        expert_ids = torch.randint(0, self.model_config.num_experts, (meta_c.shape[0], )) # FIXME: remove the dummy expert
        expert_ids = expert_ids.view((meta_c.shape[0],)).tolist()
        
        # TODO(hogura|20241201): move `get_mapping` and `permute_tokens` into CUDAGraph
        exp_mappings, _ = get_mappings_from_exp_ids(expert_ids, self.model_config.num_experts)
        hiddens = permute_tokens(hiddens, exp_mappings)
        
        if mocking:
            new_meta_c = meta_py
        else:
            new_meta_c = meta_c.to_metadata()
            new_meta_c.update_exp_ids(expert_ids, exp_mappings)
        
        # self._logger.info(f"processed batch attn {meta_c.seq_ids}")
        return hiddens, new_meta_c
    
    @nvtx_range("engine.process_batch_expert")
    def process_batch_expert_optimized(self, 
                             meta_c: Metadata, 
                             input_tensor: Tensor) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, ExpertsExecutor)
        
        # self._logger.info(f"process batch expert {meta_c.req_ids}")
        
        # NOTE: input_tensor is already permuted by expert_ids in scheduler
        # OPTIMIZE(shaoyuw): use exp_cnt to get batch_sizes
        batch_sizes = meta_c.get_expert_batch_sizes(self.model_config.num_experts)
        batch_sizes = torch.tensor(
            [batch_sizes[i] for i in self.inner_exp_rank],
            dtype=torch.int64,
            # NOTE(hogura|20241014): cuBLAS grouped_gemm requires batch_sizes to be on cpu
            device="cuda" if ENV_VARS["GROUPED_GEMM_CUTLASS"] else "cpu",
        )
        
        # self._logger.info(f"executing expert {meta_c.req_ids}")
        execution_event = torch.cuda.Event()
        if not self.model_config.enable_cuda_graph_expert:
            range_push("engine.execute")
            output = self.executor.execute(meta_c.layer_id, input_tensor, batch_sizes)
            execution_event.record()
            range_pop()
        else:
            torch.cuda.synchronize()
            
            range_push("engine.input_copy")
            num_tokens = input_tensor.shape[0]
            self.static_input[:num_tokens].copy_(input_tensor)
            self.static_input[num_tokens:].fill_(0)
            self.static_batch_sizes.copy_(batch_sizes)
            graph_id, graph_batch_size = get_graph_batch_size(num_tokens, self.graph_batch_sizes)
            range_pop()
            
            range_push("engine.graph_replay")
            self._logger.info(f"layer id {meta_c.layer_id}, graph_id {graph_id}, graph_batch_size {graph_batch_size}")
            self.graphs[meta_c.layer_id][graph_id].replay()
            output = self.static_output[:num_tokens]
            execution_event.record()
            range_pop()
            
        # 2. permute tokens back to <prefill><decode> order
        #   * NOTE(hogura|20241223): when using DP, the first sorting key is AttnId
        h2d_stream = torch.cuda.Stream()
        h2d_event = torch.cuda.Event()
                
        range_push("engine.sort_by_prefill_order")
        new_mappings = list(meta_c.sort_by_prefill_order())
        range_pop()
        
        with torch.cuda.stream(h2d_stream):
            new_mappings_cpu = torch.tensor(new_mappings, dtype=torch.int32, device="cpu", pin_memory=True)
            self.static_mappings_gpu.copy_(new_mappings_cpu, non_blocking=True)
            h2d_event.record(h2d_stream)

        range_push("engine.h2d_event")
        h2d_event.wait(h2d_stream)
        range_pop()
        
        output = permute_tokens(output, self.static_mappings_gpu)
        meta_c.update_exp_ids([], [])
        meta_c.step_layer()

        # self._logger.info(f"processed batch expert {meta_c.req_ids}")
        return output, meta_c
    
    @nvtx_range("engine.process_batch_expert")
    def process_batch_expert(self, 
                             meta_c: Metadata, 
                             input_tensor: Tensor) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, ExpertsExecutor)
        
        # self._logger.info(f"process batch expert {meta_c.req_ids}")
        
        # NOTE: input_tensor is already permuted by expert_ids in scheduler
        # OPTIMIZE(shaoyuw): use exp_cnt to get batch_sizes
        batch_sizes = meta_c.get_expert_batch_sizes(self.model_config.num_experts)
        batch_sizes = torch.tensor(
            [batch_sizes[i] for i in self.inner_exp_rank],
            dtype=torch.int64,
            # NOTE(hogura|20241014): cuBLAS grouped_gemm requires batch_sizes to be on cpu
            device="cuda" if ENV_VARS["GROUPED_GEMM_CUTLASS"] else "cpu",
        )
        
        # self._logger.info(f"executing expert {meta_c.req_ids}")
        if not self.model_config.enable_cuda_graph_expert:
            output = self.executor.execute(meta_c.layer_id, input_tensor, batch_sizes)
        else:
            torch.cuda.synchronize()
            num_tokens = input_tensor.shape[0]
            self.static_input[:num_tokens].copy_(input_tensor)
            self.static_input[num_tokens:].fill_(0)
            self.static_batch_sizes.copy_(batch_sizes)
            graph_id, graph_batch_size = get_graph_batch_size(num_tokens, self.graph_batch_sizes)
            self.graphs[meta_c.layer_id][graph_id].replay()
            output = self.static_output[:num_tokens]
        # 2. permute tokens back to <prefill><decode> order
        #   * NOTE(hogura|20241223): when using DP, the first sorting key is AttnId
        new_mappings = list(meta_c.sort_by_prefill_order())
        output = permute_tokens(output, new_mappings)
        meta_c.update_exp_ids([], [])
        meta_c.step_layer()

        # self._logger.info(f"processed batch expert {meta_c.req_ids}")
        return output, meta_c

    @nvtx_range("Engine.post_process")
    def post_process(self, output: Tensor, meta: Metadata) -> None:
        assert not self.is_attn_worker
        batch: TensorBatch = TensorBatch_C()
        batch.data = output
        batch.metadata = meta
        self.dispatcher.put(batch, 0)

    @torch.inference_mode()
    def attn_worker_loop(self):
        self._logger.info("starting engine (attn TP worker) loop")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda:0")
        torch.cuda.set_stream(self.stream)
        while not self.end_flag:
            layer_id, input_tensor, meta = self._attn_worker_preprocess()
            if layer_id == -1:
                # terminated
                self._logger.warning("TP worker received termination signal, now exit")
                break
            num_tokens = meta.num_prefill_tokens + meta.num_decode_tokens
            positions = torch.zeros(num_tokens, dtype=torch.long, device="cuda")
            self._logger.info(f"executing attn {meta}")
            self.executor.execute(layer_id, positions, input_tensor, meta)

    @torch.inference_mode()
    def loop(self):
        self._logger.info("starting engine loop")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda:0")
        torch.cuda.set_stream(self.stream)
        disagmoe_recorder_create()
        while not self.end_flag:
            # self.scheduler.wait_for_new_requests()  # !NOTE(hogura|20241008): will block this process!
            batch_info = self.scheduler.schedule() # using non-blocking schedule
            if batch_info.data is None:
                continue
            
            pool_snapshot = self.scheduler.get_pool_snapshot()
            batch_size = batch_info.data.shape[0]
            layer_id = batch_info.metadata.layer_id
            step_start_timestamp_ms = time.time() * 1000
            
            self._batch_sizes.append(batch_size)
            batch = TensorBatch.from_c(batch_info)

            meta: Metadata = batch.metadata
            output, meta = self._process_batch(meta, batch.data)
            self.post_process(output, meta)
            
            step_end_timestamp_ms = time.time() * 1000
            
            pool_snapshot_dict = dict()
            for i, size in enumerate(pool_snapshot):
                if size <= 0: 
                    continue
                layer = self.model_config.layer_ids[i]
                pool_snapshot_dict[layer] = size
                
            self._step_stats.append(
                StepInfo(step_start_timestamp_ms, 
                         step_end_timestamp_ms, 
                         batch_size, layer_id,
                         pool_snapshot_dict)
            )
        # self.recorder_output = disagmoe_recorder_output()
    
    def fetch_step_stats(self) -> Tuple[List[StepInfo], Dict[int, List[TraceContext]]]:
        """
            return: step_stats, profile_contexts
        """
        from disagmoe_c import TraceContext as TraceContext_C
        
        output: Dict[int, List[TraceContext_C]] = disagmoe_recorder_output()
        result = {}
        for key in output:
            result[key] = [TraceContext.from_c(c) for c in output[key]]
        
        # if self.engine_type in [EngineType.ATTENTION, EngineType.EXPERT]:
        #     while self.recorder_output is None:
        #         time.sleep(0.1)
        #     for key in self.recorder_output:
        #         result[key] = [TraceContext.from_c(c) for c in self.recorder_output[key]]
        
        return self._step_stats, result
    
    def release_seqs(self, seq_ids: List[int]):
        # TODO(optimize): master should only send release request to the driver
        if self.is_attn and (not self.is_attn_driver) and (not self.model_config.tp_enable_inter_group):
            # is a worker and enabled intra-group communication, no kv cache to be released.
            return
        # NOTE: due to DP, some seqs may not be in the decode_seq_lens
        seq_ids = [i for i in seq_ids if i in self.decode_seq_lens]
        self._logger.info(f"releasing seqs {seq_ids}")
        for i in seq_ids:
            # NOTE: single read/write to python dict is thread-safe due to GIL, but iterating should be protected by a lock
            self.decode_seq_lens.pop(i)
        self.block_mgr.batch_release(seq_ids)
    
    def terminate(self):
        self.end_flag = True
        if self._intra_group_tp_enabled and self.is_attn_driver:
            # sending termination signal to TP workers
            self._logger.info("TP driver sending termination signal to TP workers")
            self.buffer_meta[0] = -1
            torch.cuda.synchronize()
            dist.broadcast(self.buffer_meta, 0)
        
    def get_node_ip(self) -> str:
        return get_ip()
    
    def start_profile(self, profile_dir=None):
        assert self.device_id is not None, "Engine should be assigned with a device before profiling"
        
        if profile_dir is None:
            self._logger.info("profiling directory not specified, using default")
            profile_dir = os.environ.get("DMOE_PROFILE_DIR", f"torch_profile")
            
        self._logger.info(f"enable profiler, results stored at {profile_dir}")
    
        self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=profile_dir, 
                    worker_name=f"engine-{self.device_id}",
                    use_gzip=True,))
        self.profiler.start()
    
    def stop_profile(self):
        assert self.profiler is not None, "torch rofiler is not enabled"
        self.profiler.stop()

class SamplerEngine(Engine):
    
    def __init__(self):
        super().__init__(None, None, None, SAMPLER_DEV_ID)
        self.sampler: Sampler = None
        self.max_output_len = -1
        
    def init_core(
            self,
            layer_ids: List[int],
            # P2P Channels
            in_device_ids: List[int],
            out_device_ids: List[int],
            out_channel_infos: List[ChannelInfo],
            # Group Channels
            in_nccl_ids: Dict[int, int],
            out_device_group_ids: Dict[int, List[int]],
            out_nccl_ids: Dict[int, int],
            device_group_ids: List[int] = None,
            group_nccl_ids: str = "",
            expert_ranks: List[Tuple[int, int, int]] = [],
            local_attn_dp_rank: int = 0,
        ):
        self.sampler = init_sampler(
            self.device_id,
            self.max_output_len,
            ParallelConfig.from_c(
                1, 1, self.model_config.dp_size, 1, []
            ),
            in_device_ids,
            out_device_ids,
            [info.to_c() for info in out_channel_infos],
        )
        self._logger.info("inited sampler")
        self._t_start = time.time()
        
    def start(self):
        self.sampler.start()
        
    def fetch_finished_results(self) -> List[SloStat]:
        # convert c++ vector to python list
        results = self.sampler.fetch_finished_slo_stats()
        # if len(results) > 0:
        #     self._logger.info(f"Python sampler: fetch_finished_results: {len(results)}")
        return [SloStat.from_c(r) for r in results]
    
    def set_sampling_params(self, max_output_len: int):
        self.max_output_len = max_output_len
        
    def wait_for_n_requests(self, n_request) -> Dict[int, SloStat]:
        result = self.sampler.wait_slo_stats(n_request)
        while len(result) == 0:
            # NOTE(hogura|20241022): wait_slo_stats will return len=0 until #request==n_reqquest
            result = self.sampler.wait_slo_stats(n_request)
        new_res = {
            req_id: SloStat.from_c(stat) for req_id, stat in result.items()
        }
        return new_res
        
class TokenizerEngine(Engine):
    
    def __init__(self):
        super().__init__(None, None, None, TOKENIZER_DEV_ID)
        self.tokenizer: Tokenizer = None
        
    def process_request(self, req_id: int, input_len: int, dp_rank: int):
        # req_id (or seq_id) must > 0
        assert req_id > 0
        # TODO(hogura|20241008): only #prefill = 1 now
        assert input_len == 1
        shape = (input_len, self.model_config.hidden_size)
        # TODO(hogura|20241008): add a py-tokenizer here
        x = torch.zeros(size=shape).type(self.model_config.dtype)
        # self._logger.info("tokenizer put 1 request")
        self.tokenizer.put_request(req_id, x, dp_rank)
        
    def put_single_request(self, req_id: int, input_len: int, dp_rank: int):
        self.process_request(req_id, input_len, dp_rank)
        
    def put_requests(self, req_ids: List[int], input_lens: List[int], dp_ranks: List[int]):
        for req_id, input_len, dp_rank in zip(req_ids, input_lens, dp_ranks):
            self.process_request(req_id, input_len, dp_rank)
        
    def init_core(
            self,
            layer_ids: List[int],
            # P2P Channels
            in_device_ids: List[int],
            out_device_ids: List[int],
            out_channel_infos: List[ChannelInfo],
            # Group Channels
            in_nccl_ids: Dict[int, int],
            out_device_group_ids: Dict[int, List[int]],
            out_nccl_ids: Dict[int, int],
            device_group_ids: List[int] = None,
            group_nccl_ids: str = "",
            expert_ranks: List[Tuple[int, int, int]] = [],
            local_attn_dp_rank: int = 0,
        ):
        self.tokenizer = init_tokenizer(
            self.device_id,
            ParallelConfig.from_c(
                1, 1, self.model_config.dp_size, 1, []
            ),
            out_device_ids,
            [info.to_c() for info in out_channel_infos],
        )
        self._logger.info("inited tokenizer")
    
    def start(self):
        self.tokenizer.start()