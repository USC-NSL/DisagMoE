import torch
import time
import enum
import os

from disagmoe.executor.executor import Executor, ExpertsExecutor, AttnExecutor, CUDAGraphAttnExecutor
from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.frontend.adapter import Scheduler, MuDispatcher, Sampler, Tokenizer, BlockManager
from disagmoe.frontend.datatypes import (Metadata, ChannelInfo, TensorBatch,
                                         AttentionBatchMetadata, SloStat, TraceContext,
                                         SamplerStepInfo)
from disagmoe.frontend.ray_helper import InitCoreArgs
from disagmoe.ops.memory import permute_tokens_cuda as permute_tokens, get_mappings_from_exp_ids
from disagmoe.utils.logger import get_logger
from disagmoe.utils.utils import (get_ip, get_nccl_url_from_uid, time_ms, Timer,
                                  make_seqlens_cuda_tensor, get_graph_batch_size, StepInfo, 
                                  nvtx_range, range_push, range_pop, CudaRangeEvent)
from disagmoe.utils.metrics import Metric
from disagmoe.utils.constants import *
from disagmoe.utils.placement import ParallelConfig
from disagmoe.models.utils import make_dummy_meta
from disagmoe.models.distributed import set_tensor_model_parallel_config, set_tensor_model_parallel_channel, group_sync
from disagmoe.env import ENV_VARS

from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from typing import Optional, List, Dict, Callable, Tuple
from threading import Thread

from torch import Tensor

import torch.distributed as dist

from disagmoe_c import (init_engine, init_engine_colocate, start_engine, init_sampler, init_tokenizer, set_hosts, prepare_batch_infos,
                        TensorBatch as TensorBatch_C,
                        BlockManager as BlockManager_C,
                        recorder_create as disagmoe_recorder_create,
                        recorder_output as disagmoe_recorder_output)

class EngineType(enum.Enum):
    ATTENTION = enum.auto()
    EXPERT = enum.auto()
    HYBRID = enum.auto()
    TOKENIZER = enum.auto()
    SAMPLER = enum.auto()

class Engine:

    def __init__(self, 
                 scheduler: Optional[Scheduler] = None, 
                 executor: Optional[Executor] = None, 
                 dispatcher: Optional[MuDispatcher] = None, 
                 device_id: Optional[int] = None):
        
        assert executor is None, "Executor is initialization should be done in setup_engine"
        assert scheduler is None, "Scheduler is initialization should be done in setup_engine"
        assert dispatcher is None, "Dispatcher is initialization should be done in setup_engine"
        
        self.device_id = device_id
        self.scheduler: Scheduler = None
        self.executor: Executor = None
        self.dispatcher: MuDispatcher = None
        self.attn_scheduler: Scheduler = None
        self.expert_scheduler: Scheduler = None
        self.attn_executor: AttnExecutor = None
        self.expert_executor: ExpertsExecutor = None
        self.attn_dispatcher: MuDispatcher = None
        self.expert_dispatcher: MuDispatcher = None
        
        self.end_flag = False
        self.engine_type: EngineType = None
        self.model_config: ModelConfig = None
        self.cache_config: CacheConfig = None
        
        if device_id is not None:
            self._logger = get_logger(f"engine{device_id}")
            
        self.loop_thread = None
        
        self._process_batch: Callable = None
        
        self.block_mgr: BlockManager = None
        
        self.decode_seq_lens = {}
        self.profiler = None
        self.inner_exp_rank = []
        self.device_group_ids = []
        self.handles = []
        self.rank_in_group = 0 # EP rank in expert worker, TP rank in attention worker
        
        # for stats usage
        self._step_stats = []
        self._metric = Metric()
        self._timer = Timer()
        self._queueing_timer = {} # placeholder, not used at the moment
        self._queueing_delays = []

    @property
    def has_attn(self):
        return self.engine_type == EngineType.ATTENTION or self.engine_type == EngineType.HYBRID
    
    @property
    def has_expert(self):
        return self.engine_type == EngineType.EXPERT or self.engine_type == EngineType.HYBRID
    
    @property
    def is_attn_driver(self):
        return self.has_attn and (self._inter_group_tp_enabled or self.rank_in_group == 0)
    
    @property
    def is_attn_worker(self):
        return self._intra_group_tp_enabled and self.rank_in_group > 0
    
    @property
    def _tp_enabled(self):
        return self.has_attn and self.model_config.tp_size > 1
    
    @property
    def _inter_group_tp_enabled(self):
        return self._tp_enabled and self.model_config.tp_enable_inter_group
    
    @property
    def _intra_group_tp_enabled(self):
        return self._tp_enabled and (not self.model_config.tp_enable_inter_group)
    
    def _delegate_modules(self):
        if self.has_attn and self.has_expert:
            return
        
        if self.has_attn:
            self.scheduler = self.attn_scheduler
            self.executor = self.attn_executor
            self.dispatcher = self.attn_dispatcher
            self._process_batch = self.process_batch_attn
        elif self.has_expert:
            self.scheduler = self.expert_scheduler
            self.executor = self.expert_executor
            self.dispatcher = self.expert_dispatcher
            self._process_batch = self.process_batch_expert
        else:
            assert False, "No engine type is set"
            
    def _log_memory_usage(self, prefix: str = ""):
        free_memory, total_memory = torch.cuda.mem_get_info()
        self._logger.info(f"{prefix} CUDA free memory: {free_memory / (1024 ** 3):.2f} GB, "\
                            f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
    
    def _build_executor(self):
        if self.has_expert:
            self.expert_executor = ExpertsExecutor(self.model_config)
            # prepare inner exp rank, [n_exp_per_rank * rank, (rank + 1) * n_exp_per_rank) -> [0, n_exp_per_rank)
            self.inner_exp_rank = [0 for _ in range(self.model_config.num_experts_per_rank)]
            for i in range(self.model_config.num_experts_per_rank):
                self.inner_exp_rank[i] = self.model_config.num_experts_per_rank * self.rank_in_group + i
        if self.has_attn:
            self.attn_executor = AttnExecutor.build(self.model_config, self.cache_config)
            if self.cache_config.num_gpu_blocks is None:
                self.cache_config.num_gpu_blocks = self.determine_kv_cache_blocks() // len(self.model_config.layer_ids)
                self._logger.info(f"kv cache num_gpu_blocks: {self.cache_config.num_gpu_blocks}")
            self.attn_executor.initialize_cache(self.cache_config.num_gpu_blocks)
            self._log_memory_usage("After initialize cache")
            self.cache_config.num_gpu_blocks -= self.cache_config.num_reserved_blocks
            self.block_mgr = BlockManager_C(
                self.cache_config.block_size, 
                self.cache_config.num_gpu_blocks, 
                self.cache_config.num_reserved_blocks
            )
            if self._intra_group_tp_enabled:
                self._create_broadcast_buffers()
                
        self._warmup()
        if self.has_attn and self.model_config.enable_cuda_graph_attn:
            self.cuda_graph_executor = CUDAGraphAttnExecutor(self.model_config, self.cache_config, self.attn_executor)
            self.cuda_graph_executor.create_cuda_graph_buffers()
            self.cuda_graph_executor.capture()
            self._log_memory_usage("After build CUDA graphs")
            
        self._logger.info("Executors built")
        
    def init_core(self, core_args: InitCoreArgs):
        """
        NOTE(hogura|20241003): When using ray, all the device_id called to CUDA should become 0
        """
        disagmoe_recorder_create()
        
        self.device_group_ids = core_args.device_group_ids
        
        assert not self.model_config.tp_enable_inter_group, "TP inter group is deprecated"
        
        self.model_config.layer_ids = core_args.layer_ids
            
        self._logger.info(f"launching core: {core_args.layer_ids, core_args.in_device_ids, \
                          core_args.out_device_ids, core_args.out_channel_infos, \
                          core_args.device_group_ids, core_args.expert_ranks, core_args.local_attn_dp_rank}")
        
        # self._logger.info(f"launching core: {core_args.in_nccl_ids, core_args.out_nccl_ids, core_args.group_nccl_ids}")
        
        if self.engine_type == EngineType.HYBRID:
            self.attn_scheduler, self.attn_dispatcher, self.expert_scheduler, self.expert_dispatcher = init_engine_colocate(
                self.device_id,
                self.model_config.top_k,
                self.has_attn,
                self.has_expert,
                ParallelConfig.from_c(
                    self.model_config.tp_size if self.model_config.tp_enable_inter_group else 1, # control the init of attn_scheduler
                    self.model_config.ep_size,
                    self.model_config.dp_size,
                    self.model_config.num_experts_per_rank,
                    core_args.expert_ranks,
                ), # parallel config
                core_args.layer_ids,
                # P2P Channels
                core_args.in_device_ids,
                core_args.out_device_ids,
                [info.to_c() for info in core_args.out_channel_infos],
                # Group Channels
                core_args.in_nccl_ids,
                core_args.out_nccl_ids,
                core_args.in_nccl_ids_ext,
                core_args.out_nccl_ids_ext,
                [], # device_grou_ids, now is actually deprecated
                core_args.local_attn_dp_rank,
            )
        else:
            self.attn_scheduler, self.attn_dispatcher, self.expert_scheduler, self.expert_dispatcher = init_engine(
                self.device_id,
                self.model_config.top_k,
                self.has_attn,
                self.has_expert,
                ParallelConfig.from_c(
                    self.model_config.tp_size if self.model_config.tp_enable_inter_group else 1, # control the init of attn_scheduler
                    self.model_config.ep_size,
                    self.model_config.dp_size,
                    self.model_config.num_experts_per_rank,
                    core_args.expert_ranks,
                ), # parallel config
                core_args.layer_ids,
                # P2P Channels
                core_args.in_device_ids,
                core_args.out_device_ids,
                [info.to_c() for info in core_args.out_channel_infos],
                # Group Channels
                core_args.in_nccl_ids,
                core_args.out_nccl_ids,
                core_args.device_group_ids,
                core_args.local_attn_dp_rank,
            )
            
        if self.model_config.tp_enable_inter_group:
            set_tensor_model_parallel_channel(self.attn_scheduler.get_channel() if self.attn_scheduler is not None else None)
        else:
            if self.has_attn and self._intra_group_tp_enabled:
                dist.init_process_group(backend="nccl", 
                                        world_size=len(self.device_group_ids), 
                                        rank=self.rank_in_group,
                                        init_method=f"tcp://{get_nccl_url_from_uid(core_args.group_nccl_ids[0])}")
        
        if self.has_attn:
            self.attn_scheduler.set_max_batch_size(self.attn_max_batch_size)
        
        if self.has_expert:
            self.expert_scheduler.set_max_batch_size(self.expert_max_batch_size)
            self.static_mappings_gpu = torch.zeros((self.expert_max_batch_size, ), dtype=torch.int64, device="cuda")
            
        self._build_executor()
        
        self._delegate_modules()
        
        self._logger.info("core launched")
    
    def start(self):
        # attention TP is deprecated
        # if self.is_attn_worker:
        #     self.loop_thread = Thread(target=self.attn_worker_loop)
        start_engine(self.attn_scheduler, self.attn_dispatcher, self.expert_scheduler, self.expert_dispatcher)
        
        if self.engine_type == EngineType.HYBRID:
            self.loop_thread = Thread(target=self.dual_module_loop)
        else:
            self.loop_thread = Thread(target=self.single_module_loop)
            
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
        if engine_type in [EngineType.ATTENTION, EngineType.EXPERT, EngineType.HYBRID]:
            torch.set_default_device("cuda:0")
            stream = torch.cuda.Stream(priority=-1)
            torch.cuda.set_stream(stream)
            self._logger.info(f"set stream {stream}")
            self.stream = stream
            self.h2d_stream = torch.cuda.Stream(priority=-1)
            self.d2h_stream = torch.cuda.Stream(priority=-1)
            self.stream_schedule = torch.cuda.Stream(priority=-1)
            set_tensor_model_parallel_config(model_config)
            self._log_memory_usage("Setup device")
            free_memory, _ = torch.cuda.mem_get_info()
            self.init_gpu_memory = free_memory
            
        self.engine_type = engine_type
        self.model_config = model_config
        self.cache_config = cache_config
        
        if self.has_attn:
            self.attn_max_batch_size = model_config.max_batch_size_attn 
            
        if self.has_expert:
            self.expert_max_batch_size = model_config.max_batch_size_expert
        
        self._logger.info(f"engine setup. {self.engine_type, model_config}")
    
    def get_configured_kv_cache_blocks(self) -> int:
        return self.cache_config.num_gpu_blocks
    
    def determine_kv_cache_blocks(self) -> int:
        assert isinstance(self.attn_executor, AttnExecutor)
        torch.cuda.empty_cache()
        
        self._log_memory_usage("After allocate parameters")
        
        self.attn_executor.profile_execute(self.attn_max_batch_size)      
        torch.cuda.synchronize()  
        
        self._log_memory_usage("After profile run")
        
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = self.init_gpu_memory - free_gpu_memory
        cache_block_size = self.model_config.hidden_size // self.model_config.num_heads \
                            * self.model_config.num_kv_heads * self.cache_config.block_size * 2 * 2 # 2 for kv, 2 for fp16/bf16
        
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        
        return num_gpu_blocks
            
    def _warmup(self):
        if self.has_attn:
            self._warmup_attn()
        
        if self.has_expert:
            self._warmup_experts()
            
    def _warmup_attn(self):
        # a hacking to cuda graph, since `_pack_flash_attn_metadata` may have different behaviors.
        _enable_cuda_graph_attn = self.model_config.enable_cuda_graph_attn
        self.model_config.enable_cuda_graph_attn = False
        
        input = torch.zeros((self.attn_max_batch_size, self.model_config.hidden_size), device="cuda")
        positions = torch.zeros(self.attn_max_batch_size, dtype=torch.long, device="cuda")
        meta_py = make_dummy_meta(0, self.attn_max_batch_size)
        meta = self._pack_flash_attn_metadata(meta_py.to_c(), meta_py, [self.model_config.max_seq_len] * self.attn_max_batch_size, mocking=True)
        for layer_id in self.model_config.layer_ids:
            for _ in range(2):
                self.attn_executor.execute(layer_id, positions, input, meta)
        
        # restore the original setting
        self.model_config.enable_cuda_graph_attn = _enable_cuda_graph_attn

    def _warmup_experts(self):
        self._static_bs_cuda = torch.zeros((self.model_config.num_experts_per_rank, ), dtype=torch.int64, device="cuda")
        
        input = torch.zeros((self.expert_max_batch_size, self.model_config.hidden_size), device="cuda")
        batch_sizes = torch.tensor([self.expert_max_batch_size // len(self.inner_exp_rank)] * len(self.inner_exp_rank),
            dtype=torch.int64,
            # NOTE(hogura|20241014): cuBLAS grouped_gemm requires batch_sizes to be on cpu
            device="cuda" if ENV_VARS["GROUPED_GEMM_CUTLASS"] else "cpu")
        for layer_id in self.model_config.layer_ids:
            for _ in range(2):
                _ = self.expert_executor.execute(layer_id, self.expert_max_batch_size, input, batch_sizes)
            
    def _create_broadcast_buffers(self):
        self.buffer_meta = torch.zeros((BROADCAST_BUFFER_SIZE), dtype=torch.int32, device="cuda")
        self.buffer_tensor = torch.zeros((self.attn_max_batch_size, self.model_config.hidden_size), device="cuda")
        
        # [decode_seq_lens, query_start_loc, seq_start_loc, context_lens, slot_mapping, block_table]
        shape = (self.attn_max_batch_size + self.attn_max_batch_size * self.model_config.max_seq_len // self.cache_config.block_size, )
        self.buffer_attn_meta = torch.zeros(shape, dtype=torch.int32, device="cuda")
        
    def _wait_async_handles(self):
        for h in self.handles:
            h.wait()
        self.handles = []
    
    def _add_async_handle(self, handle):
        self.handles.append(handle)
        
    @nvtx_range("engine._update_block_table")
    def _update_block_table(self, meta_c: AttentionBatchMetadata, meta_py: AttentionBatchMetadata) -> List[int]:
        init_seq_ids = meta_py.seq_ids[ : meta_py.num_prefill_seqs]
        decode_seq_ids = meta_py.seq_ids
        # if the first layer in this attention worker, update block table and decode_seq_lens

        if meta_py.layer_id == self.model_config.layer_ids[0]:
            # allocate kv blocks for init seqs, update for all decoding seqs
            
            for i, seq_id in enumerate(init_seq_ids):
                self.decode_seq_lens[seq_id] = meta_py.init_prefill_lens[i]
                
            decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]

            # self._logger.info(f"update block table {meta_py.seq_ids}, {decode_seq_lens}")

            self.block_mgr.update_block_table(meta_c, decode_seq_lens)
            
            for i, seq_id in enumerate(decode_seq_ids):
                decode_seq_lens[i] += 1
                self.decode_seq_lens[seq_id] += 1
        else:
            decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]
        
        # self._logger.info(f"block table updated {meta_py.seq_ids}, {decode_seq_lens}")

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

        # print(f"meta_py {meta_py}, decode_seq_lens {self.decode_seq_lens}")
        
        # 1. prepare block table
        if not mocking:
            block_table_1d = self.block_mgr.prepare_block_table(meta_c, decode_seq_lens)
        else:
            # mocking=True when _warmup_attn
            block_table_1d = torch.zeros(
                (num_tokens + num_seqs * self.model_config.max_seq_len // self.cache_config.block_size, ), 
                dtype=torch.int32, device="cuda")

        slot_mapping_cuda = block_table_1d[-num_tokens : ].to(torch.int64)
        block_table_cuda = block_table_1d[ : -num_tokens].view(num_tokens, -1)

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        batch_infos_cuda = prepare_batch_infos(meta_c, decode_seq_lens)

        seq_lens_cuda, context_lens_cuda, seq_start_loc_cuda = \
            torch.split(batch_infos_cuda, [num_seqs, num_seqs, num_seqs + 1], dim=0)

        seq_lens = decode_seq_lens
            
        max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
        
        max_num_blocks = (max(seq_lens) - 1) // self.cache_config.block_size + 1
        assert mocking or self.model_config.enable_cuda_graph_attn or \
               max_num_blocks == block_table_cuda.shape[-1], f"block table wrong, {meta_py}, {block_table_cuda.shape}, {block_table_1d.shape}"
        
        return FlashAttentionMetadata(
            0,
            0,
            num_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
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
                0, # 1
                0, # 2
                meta_py.num_decode_tokens, # 3
                *decode_seq_lens, # 4
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
        assert False, "TP in attention is now deprecated"
        dist.broadcast(self.buffer_meta, 0)
        meta = self.buffer_meta.tolist()
        layer_id = meta[0]
        if layer_id == -1:
            # terminated
            return -1, None, None
        num_prefill_seqs = meta[1]
        num_prefill_tokens = meta[2]
        num_decode_tokens = meta[3]

        num_tokens = num_prefill_tokens + num_decode_tokens
        num_seqs = num_prefill_seqs + num_decode_tokens

        batch_size = get_graph_batch_size(num_tokens)[1] if self.model_config.enable_cuda_graph_attn else num_tokens
        
        input_tensor = self.buffer_tensor[ : num_tokens]
        self._add_async_handle(dist.broadcast(input_tensor, 0, async_op=True))
        
        if not self.model_config.enable_cuda_graph_attn:
            seq_lens = meta[4 : ]
            seq_lens_cuda = self.buffer_meta[4 : 4 + num_tokens]
            context_lens_tensor = seq_lens_cuda - 1
            seq_start_loc = make_seqlens_cuda_tensor(seq_lens)
        else:
            # extend seq_lens to batch_size
            seq_lens = meta[4 : ]
            for _ in range(batch_size - num_seqs):
                seq_lens.append(0)
            seq_lens_cuda = self.static_seq_lens[ : batch_size]
            seq_lens_cuda.copy_(self.buffer_meta[4 : 4 + batch_size])
            context_lens_tensor = self.static_context_lens[ : batch_size]
            context_lens_tensor.copy_(seq_lens_cuda - 1)
            seq_start_loc = self.static_seq_start_loc[ : batch_size + 1]
            seq_start_loc.copy_(make_seqlens_cuda_tensor(seq_lens))

        decode_seq_lens = seq_lens
        
        max_num_blocks = (max(seq_lens) - 1) // self.cache_config.block_size + 1
        # [slot_mapping, block_table]
        num_elems = num_tokens + max_num_blocks * num_tokens
        
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
            0,
            0,
            num_prefill_tokens + num_decode_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=[],
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_table_cuda,
            use_cuda_graph=self.model_config.enable_cuda_graph_attn,
        )
    
    @nvtx_range("engine.process_batch_attn")
    def process_batch_attn(self, 
                           meta_c: AttentionBatchMetadata, 
                           input_tensor: Tensor) -> Tuple[Tensor, Metadata]:
        # FIXME(shaoyuw): input tensor is sometimes zero tensor
        meta_py = AttentionBatchMetadata.from_c(meta_c)
        assert len(meta_py.seq_ids) > 0, "Scheduled batch is empty"

        num_tokens = meta_py.num_prefill_tokens + meta_py.num_decode_tokens

        if self.model_config.top_k > 1 and input_tensor.shape[0] > num_tokens:
            assert input_tensor.shape[0] == self.model_config.top_k * num_tokens, f"received {num_tokens} semantic tokens, in total{input_tensor.shape[0]} topk tokens"
            assert self.model_config.top_k == 2, "top_k > 2 is not supported yet, need specialized kernel"
            input_tensor = input_tensor[: num_tokens] + input_tensor[num_tokens :]
            meta_c.shrink_topk(self.model_config.top_k)

        # TODO(hogura|20241014): fill the real positions
        positions = torch.ones(num_tokens, dtype=torch.long, device="cuda")
        attn_meta = self._attn_driver_preprocess(meta_c, meta_py, input_tensor)

        self._timer.stop("preprocess")
        self._timer.start("execute")
        
        if not self.model_config.enable_cuda_graph_attn or num_tokens > self.attn_max_batch_size:
            # topk_weights and expert_ids: [batch_size, top_k]
            hiddens, expert_weights, expert_ids = self.attn_executor.execute(meta_py.layer_id, positions, input_tensor, attn_meta)
        else:
            range_push("engine.graph_replay")
            hiddens, expert_weights, expert_ids = self.cuda_graph_executor.run(meta_py.layer_id, positions, input_tensor, attn_meta)
            range_pop()
            
        self._timer.stop("execute")
        self._timer.start("postprocess")
        # _, reorder_ids = torch.sort(expert_ids.view(-1), stable=True)
        # hiddens = permute_tokens(hiddens, reorder_ids)
        # d2h_event = torch.cuda.Event()
        # with torch.cuda.stream(self.d2h_stream):
        #     new_meta_c = meta_c.to_metadata()
        #     if self.model_config.top_k > 1:
        #         new_meta_c.duplicate_topk(self.model_config.top_k)
        #     expert_ids_cpu = expert_ids.view(-1).to("cpu", non_blocking=True)
        #     reorder_ids_cpu = reorder_ids.view(-1).to("cpu", non_blocking=True)
        #     if self.model_config.top_k > 1:
        #         new_meta_c.topk_weights = expert_weights.view(-1).tolist()
        #     d2h_event.record(self.d2h_stream)
        # d2h_event.synchronize()
        # optimize: pass torch tensor to c++ and use it in cxx to reduce cpu
        # new_meta_c.update_exp_ids(expert_ids_cpu.tolist(), reorder_ids_cpu.tolist())
        
        new_meta_c = meta_c.to_metadata()
        if self.model_config.top_k > 1:
            new_meta_c.duplicate_topk(self.model_config.top_k)
            
        if self.model_config.top_k == 1:
            expert_ids = expert_ids.view(-1).tolist()
        else:
            assert self.model_config.top_k == 2, "top_k > 2 is not supported yet, need specialized kernel"
            expert_ids = expert_ids.view(-1).tolist()
            expert_weights = expert_weights.view(-1).tolist()
            new_meta_c.topk_weights = expert_weights
            
        exp_mappings, _ = get_mappings_from_exp_ids(expert_ids, self.model_config.num_experts)
        hiddens = permute_tokens(hiddens, exp_mappings)
        new_meta_c.update_exp_ids(expert_ids, exp_mappings)

        assert new_meta_c.shape[0] == hiddens.shape[0], f"shape mismatch: {new_meta_c.shape[0]} != {hiddens.shape[0]}"
        return hiddens, new_meta_c
    
    @nvtx_range("engine.process_batch_expert")
    def process_batch_expert(self, 
                             meta_c: Metadata, 
                             input_tensor: Tensor) -> Tuple[Tensor, Metadata]:
        # NOTE: input_tensor is already permuted by expert_ids in scheduler
        range_push("engine.copy_batch_sizes")
        # NOTE(hogura|20250101): MAGIC. calling tensor.shape[0] is 10us slower than meta_c.num_tokens()
        num_tokens = meta_c.num_tokens()
        if ENV_VARS["GROUPED_GEMM_CUTLASS"]:
            meta_c.get_expert_batch_sizes_cuda(
                self.model_config.num_experts, self.inner_exp_rank,
                self._static_bs_cuda, self.stream.cuda_stream
            )
            batch_sizes = self._static_bs_cuda
        else:
            batch_sizes = list(meta_c.get_expert_batch_sizes(self.model_config.num_experts))
            batch_sizes = torch.tensor(
                [batch_sizes[i] for i in self.inner_exp_rank],
                dtype=torch.int64, device="cuda"
            )
        range_pop()
        self._timer.stop("preprocess")
        self._timer.start("execute")
        
        # self._logger.info(f"executing expert {meta_c.req_ids}")
        output = self.expert_executor.execute(meta_c.layer_id, num_tokens, input_tensor, batch_sizes)
        
        self._timer.stop("execute")
        self._timer.start("postprocess")
        
        # 2. permute tokens back to <prefill><decode> order
        h2d_event = torch.cuda.Event()
        
        new_mappings = list(meta_c.sort_by_prefill_order())
        
        with torch.cuda.stream(self.h2d_stream):
            new_mappings_cpu = torch.tensor(new_mappings, dtype=torch.int64, device="cpu", pin_memory=True)
            
            if num_tokens > self.expert_max_batch_size:
                new_mappings_gpu = new_mappings_cpu.to("cuda", non_blocking=True)
            else:
                new_mappings_gpu = self.static_mappings_gpu[:num_tokens]
                new_mappings_gpu.copy_(new_mappings_cpu, non_blocking=True)
            if self.model_config.top_k > 1:
                topk_weights = torch.tensor(meta_c.topk_weights, dtype=torch.bfloat16, device="cuda").view(-1, 1)
            h2d_event.record(self.h2d_stream)

        h2d_event.wait(self.h2d_stream)

        if self.model_config.top_k > 1:
            output = output * topk_weights
        
        output = permute_tokens(output, new_mappings_gpu)
        meta_c.update_exp_ids([], [])
        meta_c.step_layer()

        # self._logger.info(f"expert send out layer {meta_c.layer_id}, {meta_c.req_ids}")
        return output, meta_c

    @nvtx_range("Engine.post_process")
    def post_process(self, output: Tensor, meta: Metadata, dispatcher) -> None:
        assert not self.is_attn_worker
        batch: TensorBatch = TensorBatch_C()
        batch.data = output
        batch.metadata = meta
        self._timer.stop("postprocess")
        
        range_push("Engine.stream_sync")
        self._timer.start("stream_sync")
        self.stream.synchronize()
        self._timer.stop("stream_sync")
        range_pop()
        dispatcher.put(batch, 0)

    @torch.inference_mode()
    def attn_worker_loop(self):
        assert False, "TP in attention is now deprecated"
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
            positions = torch.ones(num_tokens, dtype=torch.long, device="cuda")
            self._logger.info(f"executing attn {meta}")
            self.attn_executor.execute(layer_id, positions, input_tensor, meta)

    def stats_pre_process(self, batch: TensorBatch):
        self._pool_snapshot = self.scheduler.get_pool_snapshot()
        self._step_start_timestamp_ms = time_ms()
        
    def record_empty_step(self):
        if not self.model_config.enable_trace:
            return
        
        step_end_timestamp_ms = time_ms()
        self._step_stats.append(
            StepInfo(self._step_start_timestamp_ms, 
                    step_end_timestamp_ms, 
                    0, -1, -1, 
                    {x: 0 for x in self.model_config.layer_ids})
        )
        
    def stats_post_process(self, batch: TensorBatch):
        step_end_timestamp_ms = time_ms()
        self._metric.update("t_step", step_end_timestamp_ms - self._step_start_timestamp_ms)
        
        factor = self.model_config.top_k if self.has_attn else 1
        real_batch_size = batch.data.shape[0] // factor # excluding topk tokens
        if self.model_config.enable_trace:
            pool_snapshot_dict = dict()
            queueing_tokens = 0
            queueing_batches = 0
            for i, size in enumerate(self._pool_snapshot):
                if size <= 0:
                    continue
                layer = self.model_config.layer_ids[i]
                pool_snapshot_dict[layer] = size
                queueing_tokens += size
                queueing_batches += 1
                
            executed_layer_id = batch.metadata.layer_id
            if self.has_expert:
                executed_layer_id -= 1
            self._step_stats.append(
                StepInfo(self._step_start_timestamp_ms, 
                        step_end_timestamp_ms, 
                        real_batch_size, executed_layer_id,
                        self.executor.layer_mappings[executed_layer_id],
                        pool_snapshot_dict)
            )
        else:
            filtered_queue = [size for size in self._pool_snapshot if size > 0]
            queueing_tokens = sum(filtered_queue)
            queueing_batches = len(filtered_queue)
        
        if queueing_batches > 0:
            self._metric.update("t_postprocess", self._timer.get("postprocess"))
            self._metric.update("t_preprocess", self._timer.get("preprocess"))
            self._metric.update("t_execute", self._timer.get("execute") + self._timer.get("stream_sync"))
            self._metric.update("t_schedule", self._timer.get("schedule"))
            self._metric.update("effective_tokens", real_batch_size)
            self._metric.update("queueing_tokens", queueing_tokens - real_batch_size)
            self._metric.update("queueing_batches", queueing_batches - 1)

    @torch.inference_mode()
    def single_module_loop(self):
        self._logger.info("starting single_module_loop")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda:0")
        torch.cuda.set_stream(self.stream)
        disagmoe_recorder_create()
        
        prev_schedule_empty = True
        self._step_start_timestamp_ms = time_ms()
        while not self.end_flag:
            self._timer.start("schedule")
            batch_info = self.scheduler.schedule()
            if batch_info.data is None:
                if not prev_schedule_empty:
                    prev_schedule_empty = True
                    self._step_start_timestamp_ms = time_ms()
                continue
            
            if prev_schedule_empty:
                self.record_empty_step()
                prev_schedule_empty = False
            
            self._queueing_delays.append(self.scheduler.get_cur_queueing_delay())
            self._metric.step()
            
            range_push("Engine.schedule_stream_sync")
            self.stream.synchronize()
            range_pop()
            
            self._timer.stop("schedule")
            self._timer.start("preprocess")
            
            batch = TensorBatch.from_c(batch_info)
            meta: Metadata = batch.metadata
            
            self.stats_pre_process(batch)
            output, meta = self._process_batch(meta, batch.data)
            self.post_process(output, meta, self.dispatcher)
            self.stats_post_process(batch)
    
    def dual_module_loop(self):
        self._logger.info("starting dual_module_loop")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda:0")
        torch.cuda.set_stream(self.stream)
        disagmoe_recorder_create()
        
        def step(scheduler, processor, dispatcher):
            # self._timer.start("schedule")
            batch_info = scheduler.schedule()
            if batch_info.data is None:
                return
            self._metric.step()
        
            range_push("Engine.schedule_stream_sync")
            self.stream.synchronize()
            range_pop()
            
            # self._timer.stop("schedule")
            self._timer.start("preprocess")
            
            batch = TensorBatch.from_c(batch_info)
            meta: Metadata = batch.metadata
            
            # self.stats_pre_process(batch)
            output, meta = processor(meta, batch.data)
            self.post_process(output, meta, dispatcher)
            # self.stats_post_process(batch)
            
        while not self.end_flag:
            step(self.attn_scheduler, self.process_batch_attn, self.attn_dispatcher)
            step(self.expert_scheduler, self.process_batch_expert, self.expert_dispatcher)
            
    def fetch_step_stats(self) -> Tuple[List[StepInfo], Dict[int, List[TraceContext]], Metric]:
        """
            return: step_stats, profile_contexts, metric
        """
        from disagmoe_c import TraceContext as TraceContext_C
        
        output: Dict[int, List[TraceContext_C]] = disagmoe_recorder_output()
        result = {}
        for key in output:
            result[key] = [TraceContext.from_c(c) for c in output[key]]
        
        return self._step_stats, result, self._metric
    
    def fetch_queueing_delays(self) -> List[float]:
        return self._queueing_delays
    
    def release_seqs(self, seq_ids: List[int]):
        # TODO(optimize): master should only send release request to the driver
        if not self.has_attn:
            return
        if (not self.is_attn_driver) and (not self.model_config.tp_enable_inter_group):
            # is a worker and enabled intra-group communication, no kv cache to be released.
            return
        # NOTE: due to DP, some seqs may not be in the decode_seq_lens
        seq_ids = [i for i in seq_ids if i in self.decode_seq_lens]
        # self._logger.info(f"releasing seqs {seq_ids}")
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
            profile_dir = os.environ.get("DMOE_PROFILE_DIR", "torch_profile")
            
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
        
    def reset(self):
        # for stats usage
        self._metric = Metric()
        self._timer = Timer()
        self._step_stats.clear()
        self._queueing_timer.clear()
        self._queueing_delays.clear()
        self.release_seqs(list(self.decode_seq_lens.keys()))
        
    def set_schedule_policy(self, policy: str):
        if self.has_attn:
            self.attn_scheduler.set_schedule_policy(policy)
        if self.has_expert:
            self.expert_scheduler.set_schedule_policy(policy)
    
    # def set_schedule_block(self, step: int):
    #     if self.has_attn:
    #         self.attn_scheduler.set_schedule_block(step)
    #     if self.has_expert:
    #         self.expert_scheduler.set_schedule_block(step)
        
class SamplerEngine(Engine):
    
    def __init__(self):
        super().__init__(None, None, None, SAMPLER_DEV_ID)
        self.sampler: Sampler = None
        self.max_output_len = -1
        
    def init_core(self, core_args: InitCoreArgs,):
        self.sampler = init_sampler(
            self.device_id,
            self.min_output_len,
            self.max_output_len,
            self.model_config.top_k,
            ParallelConfig.from_c(
                1, 1, self.model_config.dp_size, 1, []
            ),
            core_args.in_device_ids,
            core_args.out_device_ids,
            [info.to_c() for info in core_args.out_channel_infos],
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
    
    def fetch_sampler_step_infos(self) -> List[SamplerStepInfo]:
        return [SamplerStepInfo.from_c(info) for info in self.sampler.fetch_step_infos()]
    
    def set_sampling_params(self, min_output_len: int, max_output_len: int):
        self.min_output_len = min_output_len
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
    
    def reset(self):
        self.sampler.reset()
        
class TokenizerEngine(Engine):
    
    def __init__(self):
        super().__init__(None, None, None, TOKENIZER_DEV_ID)
        self.tokenizer: Tokenizer = None
        self.t_submitted: Dict[int, int] = {}  # req_id -> timestamp when the request was submitted
        
    def process_request(self, req_id: int, init_prefill_len: int, dp_rank: int):
        # req_id (or seq_id) must > 0
        assert req_id > 0
        tensor_shape = (1, self.model_config.hidden_size)
        # TODO(hogura|20241008): add a py-tokenizer here
        x = torch.randn(tensor_shape).type(self.model_config.dtype)
        self._logger.info(f"tokenizer put request {req_id}")
        self.tokenizer.put_request(req_id, init_prefill_len, x, dp_rank)
        self.t_submitted[req_id] = time.time()
        
    def put_single_request(self, req_id: int, init_prefill_len: int, dp_rank: int):
        self.process_request(req_id, init_prefill_len, dp_rank)
        
    def put_requests(self, req_ids: List[int], init_prefill_lens: List[int], dp_ranks: List[int]):
        for req_id, init_prefill_len, dp_rank in zip(req_ids, init_prefill_lens, dp_ranks):
            self.process_request(req_id, init_prefill_len, dp_rank)
        
    def fetch_submitted_time(self):
        return self.t_submitted
        
    def init_core(self, core_args: InitCoreArgs,):
        self.tokenizer = init_tokenizer(
            self.device_id,
            ParallelConfig.from_c(
                1, 1, self.model_config.dp_size, 1, []
            ),
            core_args.out_device_ids,
            [info.to_c() for info in core_args.out_channel_infos],
        )
        self._logger.info("inited tokenizer")
    
    def start(self):
        self.tokenizer.start()
        
    def reset(self):
        self.t_submitted.clear()