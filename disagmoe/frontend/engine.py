import torch
import time
import enum
import os
import random
import zmq

from disagmoe.executor.executor import Executor, ExpertsExecutor, AttnExecutor
from disagmoe.config import ModelConfig, CacheConfig
from disagmoe.frontend.adapter import Scheduler, MuPool, MuDispatcher
from disagmoe.frontend.datatypes import (AttentionForwardBatch, BatchMetadata, TokenBatch,
                                         TraceContext, BatchDecodeResult, TokenizedRequest)
from disagmoe.frontend.ray_helper import InitCoreArgs
from disagmoe.ops.memory import permute_tokens_cuda as permute_tokens, get_mappings_from_exp_ids
from disagmoe.utils.logger import initialize_logger, get_logger
from disagmoe.utils.utils import (get_ip, get_nccl_url_from_uid, time_ms, Timer,
                                  make_seqlens_cuda_tensor, get_graph_batch_size, StepInfo, 
                                  nvtx_range, range_push, range_pop, CudaRangeEvent)
from disagmoe.utils.metrics import Metric
from disagmoe.utils.constants import *
from disagmoe.utils.placement import ParallelConfig
from disagmoe.models.distributed import set_tensor_model_parallel_config
from disagmoe.env import ENV_VARS
from disagmoe.block_manager.block_manager import BaseBlockManager
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from typing import Optional, List, Dict, Callable, Tuple, Any
from threading import Thread

from torch import Tensor

import torch.distributed as dist

from disagmoe_c import (init_disaggregated_engine, init_unified_engine,
                        start_engine, set_hosts,
                        TokenBatch as TokenBatch_C,
                        recorder_create as disagmoe_recorder_create,
                        recorder_output as disagmoe_recorder_output)

class EngineType(enum.Enum):
    ATTENTION = enum.auto()
    EXPERT = enum.auto()
    HYBRID = enum.auto()
    
class AttentionEngineMixin:
    
    _timer: Timer
    model_total_num_layers: int
    attn_executor: AttnExecutor
    model_config: ModelConfig
    cache_config: CacheConfig
    device: str
    req_seq_lens: Tensor
    block_mgr: BaseBlockManager
    dispatcher: MuDispatcher
    buffer_meta: Tensor
    buffer_attn_meta: Tensor
    dummy_sampler: "DummySampler"
    attn_dp_rank: int
    
    def build_attn_executor(self):
        self.attn_executor = AttnExecutor.build(self.model_config, self.cache_config, gate_profile_bytes=self.gate_profile_bytes)
        self.cache_config.num_gpu_blocks = self.attn_executor.get_num_cache_blocks()
        
        self.block_mgr = self.attn_executor.get_block_mgr()
        if self._tp_enabled:
            self._create_attn_broadcast_buffers()
            
        self.attn_executor.warmup(self.attn_max_batch_size)
            
        if self.model_config.enable_cuda_graph_attn:
            self.attn_executor.build_cuda_graph_executor()
    
    @nvtx_range("attn_engine.attn_driver_preprocess")
    def _attn_driver_preprocess(self, 
                                meta_c: BatchMetadata, 
                                batch: AttentionForwardBatch) -> FlashAttentionMetadata:
        self.block_mgr.update_block_table(meta_c, batch)
        seq_lens = batch.seq_lens
        
        if self._tp_enabled:
            # 1. broadcast necessary metadata
            bc_meta = [
                batch.layer_id, # 0
                0, # 1
                0, # 2
                batch.num_decode_tokens, # 3
                *seq_lens, # 4
            ]
            
            self.buffer_meta[ : len(bc_meta)].copy_(torch.tensor(bc_meta, dtype=torch.int32, device="cpu"))
            dist.broadcast(self.buffer_meta, 0)
            
            # 2. broadcast input tensor asynchronously
            self._add_async_handle(dist.broadcast(batch.data, 0, async_op=True))
            
        attn_meta = self.block_mgr.pack_flash_attn_metadata(meta_c, batch)

        if self._tp_enabled:
            self._wait_async_handles()
            
            # 3. broadcast attn_meta
            # [slot_mapping, block_table]
            max_num_blocks = attn_meta.block_tables.shape[-1]
            num_tokens = batch.num_prefill_tokens + batch.num_decode_tokens
            num_elems = num_tokens + max_num_blocks * num_tokens
            
            bc_attn_meta = self.buffer_attn_meta[ : num_elems]
            
            if not attn_meta.use_cuda_graph:
                bc_attn_meta[ : num_tokens].copy_(attn_meta.slot_mapping.to(torch.int32))
                bc_attn_meta[num_tokens : ].copy_(attn_meta.block_tables.view(-1))
            else:
                bc_attn_meta[ : num_tokens].copy_(
                    attn_meta.slot_mapping[ : num_tokens].to(torch.int32))
                get_logger().info(f"block_table shape: {attn_meta.block_tables.shape, num_tokens, max_num_blocks, bc_attn_meta.shape}")
                bc_attn_meta[num_tokens : ].copy_(
                    attn_meta.block_tables[ : num_tokens, : max_num_blocks].view(-1))
            
            dist.broadcast(bc_attn_meta, 0)
        
        return attn_meta
    
    @nvtx_range("attn_engine.process_batch_attn")
    def process_batch_attn(self, meta_c: BatchMetadata, input_tensor: Tensor) -> Tuple[Tensor, BatchMetadata]:
        # FIXME(shaoyuw): input tensor is sometimes zero tensor
        # get_logger().info(f"process_batch_attn: layer_id {meta_c.layer_id}, req_ids {meta_c.req_ids}, input_tensor.shape {input_tensor.shape}")

        with self._timer.range("preprocess"):
            batch = AttentionForwardBatch.build(meta_c, input_tensor)
            assert len(batch.req_ids) > 0, "Scheduled batch is empty"
            
            if batch.layer_id == 0:
                for i in range(batch.num_prefill_tokens):
                    self.record_max_output_lens(batch.req_ids[i], batch.max_output_lens[i])
            
            # TODO: consider the position of this code piece
            # It's better if this is done in the final expert layer, rather than having an extra hop to the attn worker
            if batch.layer_id == self.model_total_num_layers:
                # get_logger().info(f"sampling: layer_id {meta_c.layer_id}, req_ids {batch.seq_ids}")
                continue_ids, finish_req_ids = self.dummy_sampler.sample_once(batch.req_ids)
                continue_meta = meta_c.index_select(continue_ids)
                continue_meta.init_prefill_lens = [-1] * len(continue_ids)
                continue_meta.attn_dp_ranks = [self.attn_dp_rank] * len(continue_ids)
                # print(f"after sampling: continue ids {continue_ids}, continue meta {continue_meta.req_ids}, {continue_meta.init_prefill_lens}")
                self.release_seqs(finish_req_ids)
                
                batch_res = BatchDecodeResult(
                    req_ids=batch.req_ids,
                    token_ids=[0] * len(batch.req_ids),
                    is_eos=[True] * len(batch.req_ids)
                )
                for cont_id in continue_ids:
                    batch_res.is_eos[cont_id] = False
                self.detokenizer_socket.send_pyobj(batch_res)
                
                return input_tensor[continue_ids], continue_meta

            attn_meta = self._attn_driver_preprocess(meta_c, batch)
            positions = batch.seq_lens_tensor.to(torch.int64)

        with self._timer.range("execute"):
            hiddens, expert_weights, expert_ids = self.attn_executor.execute(batch.layer_id, positions, batch.data, attn_meta)
            
        with self._timer.range("postprocess"):
            # Deprecated optimization:
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
            new_meta_c = meta_c
            if self.model_config.top_k > 1:
                new_meta_c.duplicate_topk(self.model_config.top_k)
                
            if self.model_config.top_k == 1:
                expert_ids = expert_ids.view(-1).tolist()
            else:
                expert_ids = expert_ids.view(-1).tolist()
                expert_weights = expert_weights.view(-1).tolist()
                new_meta_c.topk_weights = expert_weights
                
            new_meta_c.exp_ids = expert_ids
            exp_mappings = new_meta_c.sort_by_expert()
            new_meta_c.attn_dp_ranks = [self.attn_dp_rank] * len(expert_ids)
            # exp_mappings, _ = get_mappings_from_exp_ids(expert_ids, self.model_config.num_experts)
            hiddens = permute_tokens(hiddens, exp_mappings)

            assert new_meta_c.shape[0] == hiddens.shape[0], f"shape mismatch: {new_meta_c.shape[0]} != {hiddens.shape[0]}"
            
        return hiddens, new_meta_c
    
    def release_seqs(self, seq_ids: List[int]):
        # TODO(optimize): master should only send release request to the driver
        if not self.has_attn:
            return
        if not self.is_attn_driver:
            # is a worker, no kv cache to be released.
            return
        self.block_mgr.release_seqs(seq_ids)
        
    def _create_attn_broadcast_buffers(self):
        assert False, "TP in attention is now deprecated"
        self.buffer_meta = torch.zeros((BROADCAST_BUFFER_SIZE), dtype=torch.int32, device="cuda")
        self.buffer_tensor = torch.zeros((self.attn_max_batch_size, self.model_config.hidden_size), device="cuda")
        
        # [decode_seq_lens, query_start_loc, seq_start_loc, context_lens, slot_mapping, block_table]
        shape = (self.attn_max_batch_size + self.attn_max_batch_size * self.model_config.max_seq_len // self.cache_config.block_size, )
        self.buffer_attn_meta = torch.zeros(shape, dtype=torch.int32, device="cuda")
    
    @nvtx_range("attn_engine.attn_worker_preprocess")
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

    @torch.inference_mode()
    def attn_worker_loop(self):
        assert False, "TP in attention is now deprecated"
        get_logger().info("starting engine (attn TP worker) loop")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda:0")
        torch.cuda.set_stream(self.stream)
        while not self.end_flag:
            layer_id, input_tensor, meta = self._attn_worker_preprocess()
            if layer_id == -1:
                # terminated
                get_logger().warning("TP worker received termination signal, now exit")
                break
            num_tokens = meta.num_prefill_tokens + meta.num_decode_tokens
            positions = torch.ones(num_tokens, dtype=torch.long, device="cuda")
            get_logger().info(f"executing attn {meta}")
            self.attn_executor.execute(layer_id, positions, input_tensor, meta)

class ExpertEngineMixin:

    _timer: Timer
    stream: torch.cuda.Stream
    _static_bs_cuda: Tensor
    expert_max_batch_size: int
    expert_executor: ExpertsExecutor
    model_config: ModelConfig
    cache_config: CacheConfig
    device: str
    
    def build_expert_executor(self):
        self.expert_executor = ExpertsExecutor(self.model_config)
        # prepare inner exp rank, [n_exp_per_rank * rank, (rank + 1) * n_exp_per_rank) -> [0, n_exp_per_rank)
        self.inner_exp_rank = [0 for _ in range(self.model_config.num_experts_per_rank)]
        for i in range(self.model_config.num_experts_per_rank):
            self.inner_exp_rank[i] = self.model_config.num_experts_per_rank * self.rank_in_group + i
        self.expert_executor.warmup(self.expert_max_batch_size)
        
    @nvtx_range("expert_engine.process_batch_expert")
    def process_batch_expert(self, meta_c: BatchMetadata, input_tensor: Tensor) -> Tuple[Tensor, BatchMetadata]:
        # NOTE: input_tensor is already permuted by expert_ids in scheduler
        # get_logger().info(f"process_batch_expert: layer_id {meta_c.layer_id}, req_ids {meta_c.req_ids}")
        with self._timer.range("preprocess"):
            range_push("engine.copy_batch_sizes")
            # NOTE(hogura|20250101): MAGIC. calling tensor.shape[0] is 10us slower than meta_c.num_tokens()
            num_tokens = meta_c.num_tokens()
            if self.model_config.enable_grouped_gemm:
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
            else:
                batch_sizes = list(meta_c.get_expert_batch_sizes(self.model_config.num_experts))
                batch_sizes = [batch_sizes[i] for i in self.inner_exp_rank]
            range_pop()
        
        with self._timer.range("execute"):
            # get_logger().info(f"executing expert {meta_c.req_ids}")
            output = self.expert_executor.execute(meta_c.layer_id, num_tokens, input_tensor, batch_sizes)
        
        # 2. permute tokens back to <prefill><decode> order
        with self._timer.range("postprocess"):
            h2d_event = torch.cuda.Event()
            new_mappings = list(meta_c.sort_by_attention())
            
            with torch.cuda.stream(self.h2d_stream):
                new_mappings_cpu = torch.tensor(new_mappings, dtype=torch.int64, device="cpu", pin_memory=True)
                
                if num_tokens > self.expert_max_batch_size:
                    new_mappings_gpu = new_mappings_cpu.to("cuda", non_blocking=True)
                else:
                    new_mappings_gpu = self.static_mappings_gpu[:num_tokens]
                    new_mappings_gpu.copy_(new_mappings_cpu, non_blocking=True)
                if self.model_config.top_k > 1:
                    topk_weights = torch.tensor(meta_c.topk_weights, dtype=torch.bfloat16, device="cuda").view(-1, 1)
                    output = output * topk_weights
                    
                h2d_event.record(self.h2d_stream)

            h2d_event.wait(self.h2d_stream)
            
            output = permute_tokens(output, new_mappings_gpu)
            meta_c.exp_ids = []
            meta_c.topk_weights = []
            meta_c.step_layer()

        # get_logger().info(f"expert send out layer {meta_c.layer_id}, {meta_c.req_ids}")
        return output, meta_c
    
class Engine(AttentionEngineMixin, ExpertEngineMixin):

    def __init__(self):
        
        self.device_id = None
        self.pool: Optional[MuPool] = None
        self.scheduler: Optional[Scheduler] = None
        self.dispatcher: MuDispatcher = None
        self.dummy_sampler: DummySampler = None
        
        self.end_flag = False
        self.engine_type: EngineType = None
        self.model_config: ModelConfig = None
        self.cache_config: CacheConfig = None
        
        self.model_total_num_layers = 0
            
        self.loop_thread = None
        
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
        
        self.attn_dp_rank = None
        self.expert_ep_rank = None
        self.gate_profile_bytes: Optional[bytes] = None
        self.tokenizer_socket = None
        self.detokenizer_socket = None

    @property
    def has_attn(self):
        return self.engine_type == EngineType.ATTENTION or self.engine_type == EngineType.HYBRID
    
    @property
    def has_expert(self):
        return self.engine_type == EngineType.EXPERT or self.engine_type == EngineType.HYBRID
    
    # Consider clean-up these tp legacies
    @property
    def is_attn_driver(self):
        return self.has_attn and self.rank_in_group == 0
    
    @property
    def is_attn_worker(self):
        return self._tp_enabled and self.rank_in_group > 0
    
    @property
    def _tp_enabled(self):
        return self.has_attn and self.model_config.tp_size > 1
    
    def build_executor(self):
        if self.has_expert:
            self.build_expert_executor()
        if self.has_attn:
            self.build_attn_executor()
            
        get_logger().info("Executors built")
        
    def init_core(self, core_args: InitCoreArgs):
        """
        NOTE(hogura|20241003): When using ray, all the device_id called to CUDA should become 0
        """
        disagmoe_recorder_create()
        
        self.attn_dp_rank = core_args.local_attn_dp_rank
        
        self.device_group_ids = core_args.device_group_ids
        
        self.model_config.layer_ids = core_args.layer_ids
            
        get_logger().info(
            "launching core: %s",
            (
                core_args.layer_ids,
                core_args.in_device_ids,
                core_args.out_device_ids,
                core_args.out_channel_infos,
                core_args.device_group_ids,
                core_args.expert_ranks,
                core_args.local_attn_dp_rank,
            ),
        )
        
        # get_logger().info(f"launching core: {core_args.in_nccl_ids, core_args.out_nccl_ids, core_args.group_nccl_ids}")
        if self.engine_type == EngineType.HYBRID:
            get_logger().info("launching unified engine")
            init_engine = init_unified_engine
        else:
            get_logger().info("launching disaggregated engine")
            init_engine = init_disaggregated_engine
            
        self.pool, self.scheduler, self.dispatcher = init_engine(
            self.device_id,
            core_args.local_attn_dp_rank,
            self.model_config.top_k,
            self.has_attn,
            self.has_expert,
            core_args.expert_wise_schedule,
            ParallelConfig.from_c(
                1, # control the init of attn_scheduler
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
        )
            
        if self.has_attn and self._tp_enabled:
            dist.init_process_group(backend="nccl", 
                                    world_size=len(self.device_group_ids), 
                                    rank=self.rank_in_group,
                                    init_method=f"tcp://{get_nccl_url_from_uid(core_args.group_nccl_ids[0])}")
        
        if self.has_attn:
            self.dummy_sampler = DummySampler(core_args.min_output_len, core_args.max_output_len)
        
        if self.has_expert:
            self.static_mappings_gpu = torch.zeros((self.expert_max_batch_size, ), dtype=torch.int64, device="cuda")
            
        self.build_executor()
        get_logger().info("core launched")
    
    def start(self):
        # attention TP is deprecated
        # if self.is_attn_worker:
        #     self.loop_thread = Thread(target=self.attn_worker_loop)
        start_engine(self.scheduler, self.dispatcher)
        
        self.loop_thread = Thread(target=self.single_module_loop)
            
        self.loop_thread.start()

    def set_transport(self, name: str):
        import disagmoe_c as c
        c.select_transport(name)

    def set_device_id(self, device_id: int):
        self.device_id = device_id
        
    def set_hosts(self, device_2_host: Dict[int, str]):
        device_2_host[self.device_id] = "0.0.0.0"
        set_hosts(os.getpid(), device_2_host)

    def setup_engine(
            self, 
            engine_type: EngineType,
            model_config: ModelConfig,
            cache_config: CacheConfig = None,
            rank: int = 0,
            tokenizer_addr: str = None,
            detokenizer_addr: str = None,
        ):
        
        if self.device_id is not None:
            initialize_logger(f"engine{self.device_id}")
        else:
            initialize_logger("engine")
        self.rank_in_group = rank
        torch.set_default_dtype(torch.bfloat16)
        if engine_type in [EngineType.ATTENTION, EngineType.EXPERT, EngineType.HYBRID]:
            self.device = "cuda:0" # only one visible devices for one worker
            torch.set_default_device(self.device)
            stream = torch.cuda.Stream(priority=-1)
            torch.cuda.set_stream(stream)
            get_logger().info(f"set stream {stream}")
            self.stream = stream
            self.h2d_stream = torch.cuda.Stream(priority=-1)
            self.d2h_stream = torch.cuda.Stream(priority=-1)
            self.stream_schedule = torch.cuda.Stream(priority=-1)
            set_tensor_model_parallel_config(model_config)
            
        self.engine_type = engine_type
        self.model_config = model_config
        self.cache_config = cache_config
        
        self.model_total_num_layers = model_config.num_layers
        
        if self.has_attn:
            self.attn_max_batch_size = model_config.max_batch_size_attn 
            
            context = zmq.Context(2)
            self.tokenizer_socket = context.socket(zmq.PULL)
            self.tokenizer_socket.connect(tokenizer_addr)
            
            self.detokenizer_socket = context.socket(zmq.PUSH)
            self.detokenizer_socket.connect(detokenizer_addr)
        
        if self.has_expert:
            self.expert_max_batch_size = model_config.max_batch_size_expert
        
        get_logger().info(f"engine setup. {self.engine_type, model_config}")

    # Accepts bytes uploaded via Ray object store and retains them for later
    # consumption by attention operators/gates.
    def load_gate_profile_bytes(self, data: bytes):
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("gate profile must be bytes")
        self.gate_profile_bytes = bytes(data)
        get_logger().info(f"Loaded gate profile bytes: {len(self.gate_profile_bytes)} bytes")
    
    def get_configured_kv_cache_blocks(self) -> int:
        return self.cache_config.num_gpu_blocks
    
    def _wait_async_handles(self):
        for h in self.handles:
            h.wait()
        self.handles = []
    
    def _add_async_handle(self, handle):
        self.handles.append(handle)

    @nvtx_range("Engine.post_process")
    def post_process(self, output: Tensor, meta: BatchMetadata) -> None:
        assert not self.is_attn_worker
        if meta.is_attention() and meta.layer_id == self.model_total_num_layers and meta.shape[0] > 0: # a hack for sampling check
            assert self.dummy_sampler is not None
            meta.layer_id = 0
            batch: TokenBatch = TokenBatch_C()
            batch.data = output
            batch.metadata = meta
            self.pool.put_batch(batch)
        else:
            batch: TokenBatch = TokenBatch_C()
            batch.data = output
            batch.metadata = meta
            range_push("Engine.stream_sync")
            with self._timer.range("stream_sync"):
                self.stream.synchronize()
            range_pop()
            self.dispatcher.put(batch, 0)

    def stats_pre_process(self, batch: TokenBatch):
        if not self.engine_type == EngineType.HYBRID:
            self._pool_snapshot = self.scheduler.get_pool_snapshot()
        else:
            # TODO: support snapshot for hybrid engine
            self._pool_snapshot = []
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
        
    def stats_post_process(self, batch: TokenBatch):
        step_end_timestamp_ms = time_ms()
        self._metric.update("t_step", step_end_timestamp_ms - self._step_start_timestamp_ms)
        
        factor = self.model_config.top_k if self.has_attn else 1
        real_batch_size = batch.data.shape[0] // factor # excluding topk tokens
        if self.model_config.enable_trace:
            pool_snapshot_dict = dict()
            queueing_tokens = 0
            queueing_batches = 0
            num_groups = len(self._pool_snapshot) // len(self.model_config.layer_ids)
            for i, size in enumerate(self._pool_snapshot):
                if size <= 0:
                    continue
                # layer = self.model_config.layer_ids[i]
                pool_snapshot_dict[i] = size
                queueing_tokens += size
                queueing_batches += 1
                
            executed_layer_id = batch.metadata.layer_id
            
            if batch.metadata.is_expert():
                executed_layer_id -= 1
                if num_groups > 1:
                    executed_layer_id = executed_layer_id * num_groups + batch.metadata.get_expert_id() % num_groups
            
            self._step_stats.append(
                StepInfo(self._step_start_timestamp_ms, 
                        step_end_timestamp_ms, 
                        real_batch_size, executed_layer_id,
                        executed_layer_id,
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
            
    def process_batch(self, meta_c: BatchMetadata, input_tensor: Tensor) -> Tuple[Tensor, BatchMetadata]:
        if meta_c.is_attention():
            return self.process_batch_attn(meta_c, input_tensor)
        elif meta_c.is_expert():
            return self.process_batch_expert(meta_c, input_tensor)
        else:
            assert False, "Invalid batch metadata"
            
    def recv_new_request(self):
        try:
            new_request: TokenizedRequest = self.tokenizer_socket.recv_pyobj(zmq.NOBLOCK)
            meta = BatchMetadata(
                shape=[1, self.model_config.hidden_size],
                dtype="bfloat16",
                layer_id=0,
                req_ids=[new_request.req_id],
                exp_ids=[0],
                topk_weights=[1.0],
                attn_dp_ranks=[self.attn_dp_rank],
                init_prefill_lens=[new_request.init_prefill_len],
                max_output_lens=[new_request.max_output_len],
            )
            batch: TokenBatch = TokenBatch_C()
            batch.data = torch.rand((1, self.model_config.hidden_size), dtype=torch.bfloat16, device=self.device)
            batch.metadata = meta.to_c()
            self.pool.put_batch(batch)
        except zmq.Again:
            pass

    @torch.inference_mode()
    def single_module_loop(self):
        get_logger().info("starting single_module_loop")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda:0")
        torch.cuda.set_stream(self.stream)
        disagmoe_recorder_create()
        
        prev_schedule_empty = True
        self._step_start_timestamp_ms = time_ms()
        while not self.end_flag:
            self._timer.start("schedule")
            self.recv_new_request()
            batch_info = self.scheduler.schedule()
            if batch_info.data is None:
                if not prev_schedule_empty:
                    prev_schedule_empty = True
                    self._step_start_timestamp_ms = time_ms()
                continue
            
            if prev_schedule_empty:
                self.record_empty_step()
                prev_schedule_empty = False
            
            self._metric.step()
            
            range_push("Engine.schedule_stream_sync")
            self.stream.synchronize()
            range_pop()
            
            self._timer.stop("schedule")
            
            batch = TokenBatch.from_c(batch_info)
            meta: BatchMetadata = batch.metadata
            
            # self.stats_pre_process(batch)
            output, meta = self.process_batch(meta, batch.data)
            self.post_process(output, meta)
            # self.stats_post_process(batch)
    
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
    
    def terminate(self):
        self.end_flag = True
        if self._tp_enabled and self.is_attn_driver:
            # sending termination signal to TP workers
            get_logger().info("TP driver sending termination signal to TP workers")
            self.buffer_meta[0] = -1
            torch.cuda.synchronize()
            dist.broadcast(self.buffer_meta, 0)
        
    def get_node_ip(self) -> str:
        return get_ip()
    
    def start_profile(self, profile_dir=None):
        assert self.device_id is not None, "Engine should be assigned with a device before profiling"
        
        if profile_dir is None:
            get_logger().info("profiling directory not specified, using default")
            profile_dir = os.environ.get("DMOE_PROFILE_DIR", "torch_profile")
            
        get_logger().info(f"enable profiler, results stored at {profile_dir}")
    
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
        self._timer.reset()
        self._step_stats.clear()
        self._queueing_timer.clear()
        self._queueing_delays.clear()
        if self.has_attn:
            self.block_mgr.reset_state()
        
    def record_max_output_lens(self, req_id: int, max_output_len: int):
        assert self.has_attn
        self.dummy_sampler.create_request(req_id, max_output_len)
        
    def set_schedule_policy(self, policy: str):
        if self.scheduler is not None:
            self.scheduler.set_schedule_policy(policy)
        else:
            raise ValueError("Scheduler is not initialized")
    
    # def set_schedule_block(self, step: int):
    #     if self.has_attn:
    #         self.attn_scheduler.set_schedule_block(step)
    #     if self.has_expert:
    #         self.expert_scheduler.set_schedule_block(step)

class DummySampler:
    
    def __init__(self, min_output_len: int, max_output_len: int):
        if max_output_len == min_output_len:
            max_output_len += 1
        self.min_output_len = min_output_len
        self.max_output_len = max_output_len
        self.req_max_output_len: Dict[int, int] = {}
        self.output_len: Dict[int, int] = {}
        
    def sample_once(self, req_ids: List[int]) -> Tuple[List[int], List[int]]:
        continue_ids = []
        finish_req_ids = []
        for i, req in enumerate(req_ids):
            assert req in self.output_len, f"req {req} not found in output_len"
            self.output_len[req] += 1
            if self.check_end(req):
                # print(f"req {req} finished with length {self.output_len[req]}")
                self.clean_request(req)
                finish_req_ids.append(req)
            else:
                continue_ids.append(i)
        return continue_ids, finish_req_ids
    
    def create_request(self, req_id: int, max_output_len: int = -1):
        if max_output_len == -1:
            max_output_len = random.randint(self.min_output_len, self.max_output_len)
        self.req_max_output_len[req_id] = max_output_len
        self.output_len[req_id] = 0

    def clean_request(self, req_id: int):
        if req_id in self.req_max_output_len:
            del self.req_max_output_len[req_id]
        if req_id in self.output_len:
            del self.output_len[req_id]
                    
    def check_end(self, req_id) -> bool:
        return self.output_len[req_id] >= self.req_max_output_len[req_id]