import torch
import time
import enum
import os
import random
import zmq

from disagmoe.executor.executor import Executor, ExpertsExecutor, AttnExecutor
from disagmoe.config import ModelConfig, CacheConfig, EngineConfig
from disagmoe.frontend.adapter import Scheduler, MuPool, MuDispatcher
from disagmoe.frontend.datatypes import (
    AttentionScheduleBatch,
    ForwardBatch, ForwardResult, 
    AttentionForwardBatch, AttentionForwardResult, 
    ExpertForwardBatch, ExpertForwardResult,
    BatchMetadata, TokenBatch, TokenBatchCWrapper,
    TraceContext, BatchDecodeResult, TokenizedRequest
)
from disagmoe.frontend.ray_helper import InitCoreArgs
from disagmoe.ops.memory import (
    permute_tokens_cuda as permute_tokens, 
    apply_weights_and_permute_tokens_cuda as apply_weights_and_permute_tokens,
)
from disagmoe.utils.logger import initialize_logger, get_logger
from disagmoe.frontend.profiler import EngineProfilerMixin
from disagmoe.utils.utils import (get_ip, get_nccl_url_from_uid, time_ms, Timer,
                                  make_seqlens_cuda_tensor, get_graph_batch_size, StepInfo, 
                                  nvtx_range, range_push, range_pop, CudaRangeEvent, sync_event_timeout)
from disagmoe.utils.metrics import Metric
from disagmoe.utils.constants import *
from disagmoe.utils.placement import ParallelConfig
from disagmoe.utils.utils import _log_memory_usage
from disagmoe.models.distributed import set_tensor_model_parallel_config
from disagmoe.models.experts import (
    MoEExpertsCUTLASS,
    MoEExpertsCUTLASSFP8,
    MoEExpertsDeepGemmBF16,
    MoEExpertsDeepGemmFP8,
)
from disagmoe.env import ENV_VARS
from disagmoe.block_manager.block_manager import BaseBlockManager
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from typing import Optional, List, Dict, Callable, Tuple, Any, Deque
from threading import Thread
from torch import Tensor
from collections import deque
import torch.distributed as dist

from disagmoe_c import (init_disaggregated_engine, init_unified_engine,
                        start_engine, set_hosts,
                        BatchMetadata as BatchMetadata_C,
                        TokenBatch as TokenBatch_C,
                        recorder_create as disagmoe_recorder_create,
                        recorder_output as disagmoe_recorder_output)

from disagmoe.frontend.engine_utils import EngineType, set_global_engine_config
from disagmoe.utils.gdr_context import use_gdrcopy_optimization, GdrDoubleBuffer
    
class AttentionEngineMixin:
    
    _timer: Timer
    model_total_num_layers: int
    attn_executor: AttnExecutor
    model_config: ModelConfig
    engine_config: EngineConfig
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
            
        self.attn_executor.warmup(self.engine_config.max_batch_size_attn)

        if self.engine_config.enable_cuda_graph_attn:
            self.attn_executor.build_cuda_graph_executor()
            
        self.req_tracker: Dict[int, int] = {}
        
        post_process_max_num_tokens = self.engine_config.max_batch_size_attn * self.model_config.top_k
        if use_gdrcopy_optimization:
            self.attn_token_mapping_gdr = GdrDoubleBuffer(post_process_max_num_tokens, dtype=torch.int32, device="cuda")
            self.attn_topk_weights_staging_gdr = GdrDoubleBuffer(post_process_max_num_tokens, dtype=torch.float32, device="cuda")
            self.attn_topk_ids_staging_gdr = GdrDoubleBuffer(post_process_max_num_tokens, dtype=torch.int32, device="cuda")
            self.sample_continue_ids_gdr = GdrDoubleBuffer(self.engine_config.max_batch_size_attn, dtype=torch.int64, device="cuda")
        else:
            self.attn_token_mapping_gdr = None
            self.attn_topk_weights_staging_gdr = None
            self.attn_topk_ids_staging_gdr = None
            self.sample_continue_ids_gdr = None
        
    @nvtx_range("attn_engine.attn_driver_preprocess")
    def _attn_driver_preprocess(
        self, 
        meta_c: BatchMetadata, 
        batch: AttentionScheduleBatch
    ) -> FlashAttentionMetadata:
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
    
    def preprocess_batch_attn(self, batch: TokenBatchCWrapper) -> Optional[AttentionForwardBatch]:
        schedule_batch = AttentionScheduleBatch.build(batch.metadata, batch.data)
        if schedule_batch.layer_id == self.model_total_num_layers:
            batch_wrapper = self.sample_results(schedule_batch)
            self.pool.put_batch(batch_wrapper.to_c())
            return None
        
        attn_meta = self._attn_driver_preprocess(schedule_batch.meta_c, schedule_batch)
        positions = schedule_batch.seq_lens_tensor.to(torch.int64)
        
        if use_gdrcopy_optimization:
            expert_ids_buffer_gdr = self.attn_topk_ids_staging_gdr.get_one_handle()
            expert_weights_buffer_gdr = self.attn_topk_weights_staging_gdr.get_one_handle()
            expert_ids_buffer = expert_ids_buffer_gdr.tensor
            expert_weights_buffer = expert_weights_buffer_gdr.tensor
        else:
            expert_ids_buffer = None
            expert_ids_buffer_gdr = None
            expert_weights_buffer = None
            expert_weights_buffer_gdr = None
            
        forward_batch = AttentionForwardBatch(
            layer_id=schedule_batch.layer_id,
            data=schedule_batch.data,
            num_tokens=schedule_batch.num_tokens(),
            positions=positions,
            metadata=attn_meta,
            req_ids=schedule_batch.req_ids,
            meta_c=schedule_batch.meta_c,
            proc_func=self.execute_batch_attn,
            post_proc_func=self.postprocess_batch_attn,
            expert_ids_buffer=expert_ids_buffer,
            expert_weights_buffer=expert_weights_buffer,
            expert_ids_buffer_gdr=expert_ids_buffer_gdr,
            expert_weights_buffer_gdr=expert_weights_buffer_gdr,
        )
        return forward_batch
        
    def execute_batch_attn(self, batch: AttentionForwardBatch) -> AttentionForwardResult:
        result = self.attn_executor.execute(batch)
        sync_event = torch.cuda.Event()
        sync_event.record(self.stream)
        result.sync_event = sync_event
        return result

    def postprocess_batch_attn(self, batch: AttentionForwardBatch, result: AttentionForwardResult) -> TokenBatchCWrapper:
        if result.sync_event is not None:
            sync_event_timeout(result.sync_event)
            # result.sync_event.synchronize()
            result.sync_event = None
            
        new_meta_c = batch.meta_c
        
        topk_expanded_num_tokens = batch.num_tokens * self.model_config.top_k
        
        new_meta_c.duplicate_topk(self.model_config.top_k)
        
        if use_gdrcopy_optimization and batch.metadata.use_cuda_graph:
            expert_ids = batch.expert_ids_buffer_gdr.copy_to_host_int32(topk_expanded_num_tokens)
            expert_weights = batch.expert_weights_buffer_gdr.copy_to_host_float(topk_expanded_num_tokens)
        else:
            expert_ids = result.expert_ids.flatten().tolist()
            expert_weights = result.expert_weights.flatten().tolist()
        
        new_meta_c.exp_ids = expert_ids
        new_meta_c.topk_weights = expert_weights
        exp_mappings = new_meta_c.sort_by_expert()
        
        if use_gdrcopy_optimization:
            attn_token_mapping_gdr = self.attn_token_mapping_gdr.get_one_handle()
            attn_token_mapping_gdr.copy_from_host_int32(exp_mappings)
            token_mapping_tensor = attn_token_mapping_gdr.tensor[:len(exp_mappings)]
        else:
            token_mapping_tensor = torch.tensor(exp_mappings, dtype=torch.int32, device="cuda")
     
        new_meta_c.attn_dp_ranks = [self.attn_dp_rank] * len(expert_ids)
        hiddens = permute_tokens(result.hiddens, token_mapping_tensor)
        
        result.sync_event = torch.cuda.Event()
        result.sync_event.record(self.stream)

        return TokenBatchCWrapper(data=hiddens, metadata=new_meta_c)
    
    def sample_results(self, batch: AttentionScheduleBatch) -> TokenBatchCWrapper:
        # get_logger().info(f"sampling: layer_id {batch.meta_c.layer_id}, req_ids {batch.seq_ids}")
        continue_ids, finish_req_ids = self.dummy_sampler.sample_once(batch.req_ids)
        continue_meta = batch.meta_c.index_select(continue_ids)
        continue_meta.init_prefill_lens = [-1] * len(continue_ids)
        continue_meta.attn_dp_ranks = [self.attn_dp_rank] * len(continue_ids)
        continue_meta.layer_id = 0
        self.release_seqs(finish_req_ids)
        
        batch_res = BatchDecodeResult(
            req_ids=batch.req_ids,
            token_ids=[0] * len(batch.req_ids),
            is_eos=[True] * len(batch.req_ids)
        )
        for cont_id in continue_ids:
            batch_res.is_eos[cont_id] = False
        self.detokenizer_socket.send_pyobj(batch_res)
        
        if len(continue_ids) == 0:
            continued_data = batch.data[:0]
        elif use_gdrcopy_optimization:
            idx_gdr = self.sample_continue_ids_gdr.get_one_handle()
            idx_gdr.copy_from_host_int64(continue_ids)
            idx_tensor = idx_gdr.tensor[: len(continue_ids)]
            continued_data = torch.index_select(batch.data, 0, idx_tensor)
        else:
            continued_data = batch.data[continue_ids]
        return TokenBatchCWrapper(data=continued_data, metadata=continue_meta)
    
    @nvtx_range("attn_engine.process_batch_attn")
    def process_batch_attn(self, batch: TokenBatchCWrapper) -> Optional[TokenBatchCWrapper]:
        # get_logger().info(f"process_batch_attn: layer_id {batch.metadata.layer_id}, req_ids {batch.metadata.req_ids}, input_tensor.shape {batch.data.shape}")
        forward_batch = self.preprocess_batch_attn(batch)
        if forward_batch is None:
            return None
        result = self.execute_batch_attn(forward_batch)
        return self.postprocess_batch_attn(forward_batch, result)
    
    def release_seqs(self, seq_ids: List[int]):
        self.block_mgr.release_seqs(seq_ids)
        
    def _create_attn_broadcast_buffers(self):
        assert False, "TP in attention is now deprecated"
        self.buffer_meta = torch.zeros((BROADCAST_BUFFER_SIZE), dtype=torch.int32, device="cuda")
        self.buffer_tensor = torch.zeros((self.engine_config.max_batch_size_attn, self.model_config.hidden_size), device="cuda")
        
        # [decode_seq_lens, query_start_loc, seq_start_loc, context_lens, slot_mapping, block_table]
        shape = (self.engine_config.max_batch_size_attn + self.engine_config.max_batch_size_attn * self.model_config.max_seq_len // self.cache_config.block_size, )
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

        batch_size = get_graph_batch_size(num_tokens)[1] if self.engine_config.enable_cuda_graph_attn else num_tokens
        
        input_tensor = self.buffer_tensor[ : num_tokens]
        self._add_async_handle(dist.broadcast(input_tensor, 0, async_op=True))
        
        if not self.engine_config.enable_cuda_graph_attn:
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
        
        if not self.engine_config.enable_cuda_graph_attn:
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
            use_cuda_graph=self.engine_config.enable_cuda_graph_attn,
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
    engine_config: EngineConfig
    cache_config: CacheConfig
    device: str
    
    def build_expert_executor(self):
        if self.local_expert_ids is not None and len(self.local_expert_ids) > 0:
            self.local_to_gloabl_expert_rank = list(self.local_expert_ids)
            self.global_to_local_expert_rank = [-1 for _ in range(self.model_config.num_experts)]
            for local_idx, global_id in enumerate(self.local_expert_ids):
                self.global_to_local_expert_rank[global_id] = local_idx
        else:
            self.local_to_gloabl_expert_rank = [0 for _ in range(self.model_config.num_experts_per_rank)]
            self.global_to_local_expert_rank = [-1 for _ in range(self.model_config.num_experts)]
            for i in range(self.model_config.num_experts_per_rank):
                self.local_to_gloabl_expert_rank[i] = self.model_config.num_experts_per_rank * self.rank_in_group + i
                self.global_to_local_expert_rank[self.model_config.num_experts_per_rank * self.rank_in_group + i] = i
        self.expert_executor = ExpertsExecutor(self.model_config, self.local_to_gloabl_expert_rank, self.global_to_local_expert_rank)

        if self.engine_config.enable_cuda_graph_expert and self.expert_executor.expert_cls in [
            MoEExpertsDeepGemmBF16,
            MoEExpertsDeepGemmFP8,
            MoEExpertsCUTLASS,
            MoEExpertsCUTLASSFP8,
        ]:
            self.expert_executor.build_cuda_graph_executor()
        else:
            self.expert_executor.warmup(self.expert_max_batch_size)

        _log_memory_usage("After building expert executor")
        
        if use_gdrcopy_optimization:
            self.expert_token_mapping_gdr = GdrDoubleBuffer(self.expert_max_batch_size, dtype=torch.int32, device="cuda")
            self.expert_token_weights_staging_gdr = GdrDoubleBuffer(self.expert_max_batch_size, dtype=torch.float32, device="cuda")
        else:
            self.expert_token_mapping_gdr = None
            self.expert_token_weights_staging_gdr = None
    
    def preprocess_batch_expert(self, batch: TokenBatchCWrapper) -> Optional[ExpertForwardBatch]:
        meta_c = batch.metadata
        input_tensor = batch.data
        with self._timer.range("preprocess"):
            batch_sizes, m_indices = self.expert_executor.prepare_bsz_and_indices(meta_c)
        return ExpertForwardBatch(
            layer_id=meta_c.layer_id,
            num_tokens=meta_c.num_tokens(),
            data=input_tensor,
            batch_sizes=batch_sizes,
            m_indices=m_indices,
            meta_c=meta_c,
            proc_func=self.execute_batch_expert,
            post_proc_func=self.postprocess_batch_expert,
        )
        
    def execute_batch_expert(self, batch: ExpertForwardBatch) -> ExpertForwardResult:
        _sample = self._advanced_logger.should_sample()
        if _sample:
            _evt_start = torch.cuda.Event(enable_timing=True)
            _evt_end   = torch.cuda.Event(enable_timing=True)
            _evt_start.record()
        with self._timer.range("execute"):
            hiddens = self.expert_executor.execute(batch)
        if _sample:
            _evt_end.record()
            # Defer timing read — will be drained non-blockingly at next loop top
            self._pending_moe_events.append((batch.num_tokens, _evt_start, _evt_end))
        
        topk_weights = batch.meta_c.topk_weights
        new_mappings = list(batch.meta_c.sort_by_attention())
            
        if use_gdrcopy_optimization:
            expert_weights_buffer_gdr = self.expert_token_weights_staging_gdr.get_one_handle()
            expert_token_mapping_buffer_gdr = self.expert_token_mapping_gdr.get_one_handle()
            expert_weights_buffer_gdr.copy_from_host_float(topk_weights)
            expert_token_mapping_buffer_gdr.copy_from_host_int32(new_mappings)
            expert_weights_tensor = expert_weights_buffer_gdr.tensor
            expert_token_mapping_tensor = expert_token_mapping_buffer_gdr.tensor
        else:
            topk_weights = torch.tensor(batch.meta_c.topk_weights, dtype=torch.float32, device="cuda")
            new_mappings = torch.tensor(new_mappings, dtype=torch.int32, device="cuda")
            expert_weights_tensor = topk_weights
            expert_token_mapping_tensor = new_mappings
        
        permuted_tokens = apply_weights_and_permute_tokens(hiddens, expert_weights_tensor, expert_token_mapping_tensor)
        batch.meta_c.exp_ids = []
        batch.meta_c.topk_weights = []
        batch.meta_c.step_layer()
        
        sync_event = torch.cuda.Event()
        sync_event.record(self.stream)
        return ExpertForwardResult(hiddens=permuted_tokens, sync_event=sync_event)
        
    def postprocess_batch_expert(self, batch: ExpertForwardBatch, result: ExpertForwardResult) -> TokenBatchCWrapper:
        with self._timer.range("postprocess"):
            if result.sync_event is not None:
                sync_event_timeout(result.sync_event)
                # result.sync_event.synchronize()
                result.sync_event = None

        return TokenBatchCWrapper(data=result.hiddens, metadata=batch.meta_c)
    
    @nvtx_range("expert_engine.process_batch_expert")
    def process_batch_expert(self, batch: TokenBatchCWrapper) -> Optional[TokenBatchCWrapper]:
        # get_logger().info(f"process_batch_expert: layer_id {batch.metadata.layer_id}, req_ids {batch.metadata.req_ids}")
        try:
            forward_batch = self.preprocess_batch_expert(batch)
            result = self.execute_batch_expert(forward_batch)
            return self.postprocess_batch_expert(forward_batch, result)
        except Exception as e:
            get_logger().error(f"Exception in process_batch_expert: {e}")
            with open(f"engine-expert-{self.device_id}.err", "wt") as f:
                f.write(f"Exception in process_batch_expert: {e}\n")
                f.write(f"batch: {forward_batch.to_string()}\n")
                f.write(f"bsz: {forward_batch.meta_c.get_expert_batch_sizes(self.model_config.num_experts)}\n")
            raise e
    
class Engine(AttentionEngineMixin, ExpertEngineMixin, EngineProfilerMixin):

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
        self.engine_config: EngineConfig = None
        
        self.model_total_num_layers = 0
            
        self.loop_thread = None
        
        self.profiler = None
        self.profile_dir = None
        self.local_to_gloabl_expert_rank = []
        self.global_to_local_expert_rank = []
        self.device_group_ids = []
        self.handles = []
        self.rank_in_group = 0 # EP rank in expert worker, TP rank in attention worker
        
        # for stats usage
        self._step_stats = []
        self._metric = Metric()
        self._timer = Timer()
        self._queueing_timer = {} # placeholder, not used at the moment
        self._queueing_delays = []

        from disagmoe.utils.advanced_logger import AdvancedLogger
        self._advanced_logger = AdvancedLogger(False, "", 0)
        # Pending CUDA event pairs for non-blocking groupedGEMM timing.
        # Each entry: (batch_size, start_event, end_event)
        self._pending_moe_events: deque = deque()
        
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
        self.local_num_experts = core_args.local_num_experts
        self.local_expert_ids = core_args.local_expert_ids
        
        self.device_group_ids = core_args.device_group_ids
        
        self.model_config.layer_ids = core_args.layer_ids
            
        get_logger().debug(
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
        
        _log_memory_usage("Before launching engine")
        
        if self.engine_type == EngineType.HYBRID:
            get_logger().info("launching unified engine")
            init_engine = init_unified_engine
        else:
            get_logger().info("launching disaggregated engine")
            init_engine = init_disaggregated_engine

        self.pool, self.scheduler, self.dispatcher = init_engine(
            core_args.world_size,
            self.device_id,
            core_args.local_attn_dp_rank,
            self.model_config.top_k,
            self.has_attn,
            self.has_expert,
            core_args.expert_wise_schedule,
            ParallelConfig.to_c(
                1,
                self.model_config.ep_size,
                self.model_config.dp_size,
                self.local_num_experts if self.local_num_experts > 0 else self.model_config.num_experts_per_rank,
                core_args.expert_ranks,
                n_total_experts=self.model_config.num_experts,
            ),
            core_args.layer_ids,
            # P2P Channels
            core_args.in_device_ids,
            core_args.out_device_ids,
            core_args.inbound_nccl_ids,
            core_args.outbound_nccl_ids,
            [info.to_c() for info in core_args.out_channel_infos],
            # Unified scheduler configuration (only used for unified/colocate engine).
            self.engine_config.unified_scheduler_type,
            self.engine_config.defrag_weight_decay,
            self.engine_config.defrag_lookahead_steps,
            self.engine_config.defrag_lookback_steps,
        )
        
        self.scheduler.set_schedule_token_threshold(self.engine_config.max_batch_size_attn, self.engine_config.max_batch_size_expert)
        
        _log_memory_usage("After initializing engine")
            
        if self.has_attn and self._tp_enabled:
            dist.init_process_group(backend="nccl", 
                                    world_size=len(self.device_group_ids), 
                                    rank=self.rank_in_group,
                                    init_method=f"tcp://{get_nccl_url_from_uid(core_args.group_nccl_ids[0])}")
        
        if self.has_attn:
            self.dummy_sampler = DummySampler()
        
        if self.has_expert:
            self.static_mappings_gpu = torch.zeros((self.expert_max_batch_size, ), dtype=torch.int64, device="cuda")
            
        self.build_executor()
        get_logger().info("core launched")

        from disagmoe.utils.advanced_logger import AdvancedLogger
        self._advanced_logger = AdvancedLogger(
            self.engine_config.enable_advanced_logging,
            getattr(self.engine_config, "advanced_logging_dir", "./advanced_logs"),
            self.device_id,
            sample_rate=getattr(self.engine_config, "advanced_logging_sample_rate", 0.1),
        )
        if self._advanced_logger.enabled:
            if self.pool is not None:
                self.pool.set_tracing_enabled(True)
            if self.dispatcher is not None:
                self.dispatcher.set_tracing_enabled(True)
            try:
                import disagmoe_c as _dc
                if hasattr(_dc, "nixl_set_tracing_enabled"):
                    _dc.nixl_set_tracing_enabled(True)
            except Exception:
                pass
            import atexit, signal
            def _atexit_dump():
                try:
                    self.dump_advanced_logs("_atexit")
                except Exception as _e:
                    try: get_logger().warning(f"atexit dump failed: {_e}")
                    except Exception: pass
            atexit.register(_atexit_dump)
            def _sig_dump(_signum, _frame):
                _atexit_dump()
                os._exit(0)
            for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
                try: signal.signal(_sig, _sig_dump)
                except Exception: pass
    
    def start(self):
        # attention TP is deprecated
        # if self.is_attn_worker:
        #     self.loop_thread = Thread(target=self.attn_worker_loop)
        if hasattr(self.engine_config, 'max_pending_sends'):
            self.dispatcher.set_max_pending_sends(self.engine_config.max_pending_sends)
        start_engine(self.scheduler, self.dispatcher)
        
        self.loop_thread = Thread(target=self.single_module_loop_overlap)
            
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
            engine_config: EngineConfig,
            cache_config: CacheConfig = None,
            rank: int = 0,
            tokenizer_addr: str = None,
            detokenizer_addr: str = None,
        ):
        
        initialize_logger(f"engine{self.device_id}")
        self.rank_in_group = rank
        torch.set_default_dtype(torch.bfloat16)
        self.device = torch.device("cuda:0") # only one visible devices for one worker, set by ray
        torch.cuda.set_device(self.device)
        torch.set_default_device(self.device)
        torch.cuda.synchronize()
        self.stream = torch.cuda.Stream(device=self.device, priority=-1)
        torch.cuda.set_stream(self.stream)
        get_logger().info(f"stream: {self.stream.cuda_stream}")
        set_tensor_model_parallel_config(model_config)
            
        self.engine_type = engine_type
        self.model_config = model_config
        self.engine_config = engine_config
        self.cache_config = cache_config
        
        set_global_engine_config(engine_config)
        
        self.model_total_num_layers = model_config.num_layers
        
        if self.has_attn:
            context = zmq.Context(2)
            self.tokenizer_socket = context.socket(zmq.PULL)
            self.tokenizer_socket.connect(tokenizer_addr)
            
            self.detokenizer_socket = context.socket(zmq.PUSH)
            self.detokenizer_socket.connect(detokenizer_addr)
        
        if self.has_expert:
            self.expert_max_batch_size = engine_config.max_batch_size_expert
        
        get_logger().info(f"engine setup. {self.engine_type, engine_config}")

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
    def post_process(
        self, 
        batch: TokenBatchCWrapper,
        sync_event: Optional[torch.cuda.Event] = None,
        sync_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        assert not self.is_attn_worker

        range_push("Engine.stream_sync")
        with self._timer.range("stream_sync"):
            if sync_event is not None:
                sync_event_timeout(sync_event)
            if sync_stream is not None:
                sync_stream.synchronize()
        range_pop()
        self.dispatcher.put(batch.to_c(), 0)

    def stats_pre_process(self, batch: TokenBatch):
        self._pool_snapshot = self.scheduler.get_pool_snapshot()
        self._step_start_timestamp_ms = time_ms()
        
    def record_empty_step(self):
        if not self.engine_config.enable_trace:
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
        if self.engine_config.enable_trace:
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
            
    def preprocess_batch(self, batch: TokenBatchCWrapper) -> Optional[ForwardBatch]:
        meta_c = batch.metadata
        if meta_c.is_attention():
            return self.preprocess_batch_attn(batch)
        elif meta_c.is_expert():
            return self.preprocess_batch_expert(batch)
        else:
            assert False, "Invalid batch metadata"
            
    def process_batch(self, batch: TokenBatchCWrapper) -> Optional[TokenBatchCWrapper]:
        meta_c = batch.metadata
        if meta_c.is_attention():
            return self.process_batch_attn(batch)
        elif meta_c.is_expert():
            return self.process_batch_expert(batch)
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
            )
            self.record_max_output_lens(new_request.req_id, new_request.max_output_len)
            batch: TokenBatch = TokenBatch_C()
            batch.data = torch.rand((1, self.model_config.hidden_size), dtype=torch.bfloat16, device=self.device)
            batch.metadata = meta.to_c()
            self.pool.put_batch(batch)
            self._advanced_logger.log_pool_put(1)
        except zmq.Again:
            pass
        
    def _drain_moe_events(self):
        """Non-blockingly drain completed CUDA event pairs and log their GPU elapsed time.
        Called at the top of each loop iteration — no synchronization, no stall.
        """
        while self._pending_moe_events:
            bsz, evt_start, evt_end = self._pending_moe_events[0]
            if not evt_end.query():  # GPU hasn't passed end marker yet — bail out
                break
            elapsed_ms = evt_start.elapsed_time(evt_end)  # accurate GPU time
            self._advanced_logger.log_moe_step(bsz, elapsed_ms)
            self._pending_moe_events.popleft()

    def _drain_transport_stats(self):
        if not self._advanced_logger.enabled:
            return

        if self.pool is not None:
            for (peer_id, layer_id, num_tokens, num_bytes,
                 posted_ts_s, completed_ts_s, is_local) in self.pool.drain_recv_completion_stats():
                self._advanced_logger.log_recv_completion(
                    peer_id, layer_id, num_tokens, num_bytes,
                    posted_ts_s, completed_ts_s, is_local)

        if self.dispatcher is not None:
            for (start_ts_s, end_ts_s, pending_before,
                 max_pending, yield_count) in self.dispatcher.drain_pending_send_stall_stats():
                self._advanced_logger.log_pending_send_stall(
                    start_ts_s, end_ts_s, pending_before, max_pending, yield_count)
            if hasattr(self.dispatcher, "drain_send_msg_size_stats"):
                for (peer_cid, layer_id, num_tokens, num_bytes,
                     ts_s, transport) in self.dispatcher.drain_send_msg_size_stats():
                    self._advanced_logger.log_dispatcher_send(
                        peer_cid, layer_id, num_tokens, num_bytes, ts_s, transport)

        try:
            import disagmoe_c as _dc
            if hasattr(_dc, "nixl_drain_send_traces"):
                for trace in _dc.nixl_drain_send_traces():
                    self._advanced_logger.log_nixl_send_trace(trace)
            if hasattr(_dc, "nixl_drain_recv_traces"):
                for trace in _dc.nixl_drain_recv_traces():
                    self._advanced_logger.log_nixl_recv_trace(trace)
        except Exception:
            pass

    def _schedule_batch_with_optional_trace(self):
        if not self._advanced_logger.enabled:
            return self.scheduler.schedule(), None
        if not use_gdrcopy_optimization:
            return self.scheduler.schedule(), None
        trace = self.scheduler.schedule_trace()
        return trace.batch, trace.pool_snapshot

    @torch.inference_mode()
    def single_module_loop_overlap(self):
        # should be used with cuda graph, some concerns
        # 1. copy results out to another buffer
        get_logger().info("starting single_module_loop")
        torch.cuda.synchronize()
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(self.device)
        torch.set_default_device(self.device)
        torch.cuda.set_stream(self.stream)
        
        result_queue: Deque[Tuple[Optional[ForwardBatch], Optional[ForwardResult]]] = deque()
        forward_batch = None
        last_batch = None
        idle_conunt = 0

        _inf_iter_count = 0
        _inf_sched_ns = 0
        _inf_proc_ns = 0
        _inf_post_ns = 0
        _inf_had_batch = 0
        _inf_had_last = 0
        _inf_t_window = time.perf_counter()

        try:
            while not self.end_flag:
                _inf_iter_count += 1
                _it_t0 = time.perf_counter()
                self._drain_moe_events()
                self._drain_transport_stats()
                self.recv_new_request()
                _sched_t0 = time.perf_counter()
                batch, _trace_snapshot = self._schedule_batch_with_optional_trace()
                forward_batch = None
                if batch.data is not None:
                    _sched_ms = (time.perf_counter() - _sched_t0) * 1000.0
                    idle_conunt = 0
                    batch_wrapper = TokenBatchCWrapper.from_c(batch)
                    meta = batch_wrapper.metadata
                    _sched_ts = time.time() if self._advanced_logger.enabled else 0.0
                    if self._advanced_logger.enabled and _trace_snapshot is not None:
                        _pool_snapshot = list(_trace_snapshot)
                        _num_attn_in_pool = len(_pool_snapshot) - self.model_total_num_layers
                        _unified_layer = meta.layer_id + (_num_attn_in_pool if meta.is_expert() else 0)
                        self._advanced_logger.log_queue_snapshot(_sched_ts, _unified_layer, _pool_snapshot)
                    if self._advanced_logger.should_sample() and meta.is_expert():
                        _layer = meta.layer_id
                        _delay_ms = _sched_ms / max(1, meta.num_tokens())
                        for _eid in set(meta.exp_ids):
                            self._advanced_logger.log_queuing_delay(_layer, _eid, _delay_ms, _sched_ts)
                    _proc_t0 = time.perf_counter()
                    forward_batch = self.preprocess_batch(batch_wrapper)
                    if forward_batch is not None:
                        result = forward_batch.proc_func(forward_batch)
                        result_queue.append((forward_batch, result))
                    self.step_profile(batch.metadata.num_tokens())
                    _inf_proc_ns += int((time.perf_counter() - _proc_t0) * 1e9)
                    _inf_had_batch += 1
                    _inf_sched_ns += int((_proc_t0 - _sched_t0) * 1e9)

                _post_t0 = time.perf_counter()
                if last_batch:
                    tmp_batch, tmp_result = result_queue.popleft()
                    if tmp_batch is None:
                        pass
                    else:
                        if tmp_result.sync_event is not None:
                            sync_event_timeout(tmp_result.sync_event)
                            tmp_result.sync_event = None
                        final_result = tmp_batch.post_proc_func(tmp_batch, tmp_result)
                        self.post_process(final_result, sync_event=tmp_result.sync_event)
                    _inf_had_last += 1
                elif batch.data is None:
                    idle_conunt += 1
                _inf_post_ns += int((time.perf_counter() - _post_t0) * 1e9)

                last_batch = forward_batch

                if _inf_iter_count % 500 == 0:
                    _now = time.perf_counter()
                    _wnd = _now - _inf_t_window
                    get_logger().info(
                        f"[INF_LOOP dev={self.device_id}] iters={_inf_iter_count} "
                        f"had_batch={_inf_had_batch} had_last={_inf_had_last} idle={idle_conunt} "
                        f"avg_sched_ms={_inf_sched_ns/1e6/max(1,_inf_had_batch):.3f} "
                        f"avg_proc_ms={_inf_proc_ns/1e6/max(1,_inf_had_batch):.3f} "
                        f"avg_post_ms={_inf_post_ns/1e6/max(1,_inf_had_last):.3f} "
                        f"window_s={_wnd:.2f} "
                        f"iters/sec={500/max(_wnd,1e-9):.1f}"
                    )
                    _inf_iter_count = 0
                    _inf_sched_ns = _inf_proc_ns = _inf_post_ns = 0
                    _inf_had_batch = _inf_had_last = 0
                    _inf_t_window = _now
        except Exception as e:
            get_logger().error(f"Exception in single_module_loop_overlap: {e}")
            with open(f"engine-{self.device_id}.err", "wt") as f:
                # write this batch and last batch to file
                str1 = forward_batch.to_string() if forward_batch is not None else "None"
                str2 = last_batch.to_string() if last_batch is not None else "None"
                f.write(f"batch: {str1}\n")
                f.write(f"last_batch: {str2}\n")
            raise e
        
    @torch.inference_mode()
    def single_module_loop(self):
        get_logger().info("starting single_module_loop")
        torch.cuda.synchronize()
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(self.device)
        torch.set_default_device(self.device)
        torch.cuda.set_stream(self.stream)
        disagmoe_recorder_create()
        
        prev_schedule_empty = True
        self._step_start_timestamp_ms = time_ms()
        try:
            while not self.end_flag:
                self._timer.start("schedule")
                self._drain_transport_stats()
                self.recv_new_request()
                batch, _trace_snapshot = self._schedule_batch_with_optional_trace()
                if batch.data is None:
                    if not prev_schedule_empty:
                        prev_schedule_empty = True
                        self._step_start_timestamp_ms = time_ms()
                    continue
                
                if prev_schedule_empty:
                    self.record_empty_step()
                    prev_schedule_empty = False
                
                self._metric.step()
                
                self._timer.stop("schedule")
                
                batch_wrapper = TokenBatchCWrapper.from_c(batch)
                meta: BatchMetadata = batch_wrapper.metadata

                _sched_ts = time.time() if self._advanced_logger.enabled else 0.0
                if self._advanced_logger.enabled and _trace_snapshot is not None:
                    _pool_snapshot = list(_trace_snapshot)
                    _num_attn_in_pool = len(_pool_snapshot) - self.model_total_num_layers
                    _unified_layer = meta.layer_id + (_num_attn_in_pool if meta.is_expert() else 0)
                    self._advanced_logger.log_queue_snapshot(_sched_ts, _unified_layer, _pool_snapshot)

                if self._advanced_logger.should_sample():
                    if meta.is_expert():
                        _schedule_ms = self._timer.get("schedule")
                        _layer = meta.layer_id
                        _delay_ms = _schedule_ms / max(1, meta.num_tokens())
                        for _eid in set(meta.exp_ids):
                            self._advanced_logger.log_queuing_delay(_layer, _eid, _delay_ms, _sched_ts)
                 
                # self.stats_pre_process(batch)
                self.step_profile(meta.num_tokens())
                batch_wrapper = self.process_batch(batch_wrapper)
                if batch_wrapper is None:
                    continue
                self.post_process(batch_wrapper, sync_stream=self.stream)
                # self.stats_post_process(batch)
        except Exception as e:
            get_logger().error(f"Exception in single_module_loop: {e}")
            with open(f"engine-{self.device_id}.err", "wt") as f:
                f.write(f"Exception in single_module_loop: {e}\n")
                f.write(f"batch is attention: {meta.is_attention()}, layer_id: {meta.layer_id}\n, num_tokens: {meta.num_tokens()}\n")
            raise e
    
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

    def dump_advanced_logs(self, suffix: str = "") -> Optional[str]:
        # Flush any remaining in-flight CUDA timing events before dumping.
        # A sync here is acceptable — dump is called off the hot path.
        if self._pending_moe_events:
            torch.cuda.synchronize()
            self._drain_moe_events()
        self._drain_transport_stats()
        return self._advanced_logger.dump(suffix)

    def get_advanced_log_data(self) -> Optional[dict]:
        """Return collected advanced log data for central collection by controller."""
        # Flush any remaining in-flight CUDA timing events before collecting.
        if self._pending_moe_events:
            torch.cuda.synchronize()
            self._drain_moe_events()
        self._drain_transport_stats()
        return self._advanced_logger.get_data()
    
    def get_pool_snapshot(self) -> List[int]:
        return self.scheduler.get_pool_snapshot()
    
    def get_topk_pool_snapshot(self) -> List[int]:
        return self.scheduler.get_topk_pool_snapshot()
    
    def terminate(self):
        self.end_flag = True
        if self._tp_enabled and self.is_attn_driver:
            # sending termination signal to TP workers
            get_logger().info("TP driver sending termination signal to TP workers")
            self.buffer_meta[0] = -1
            torch.cuda.synchronize()
            dist.broadcast(self.buffer_meta, 0)
        
    def get_node_ip(self, host_ifname: str = "") -> str:
        return get_ip(host_ifname)

    def get_worker_identity(self, host_ifname: str = "") -> Dict[str, str]:
        import ray

        runtime_ctx = ray.get_runtime_context()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cuda_device = ""
        if len(cuda_visible) > 0:
            cuda_device = cuda_visible.split(",")[0].strip()
        else:
            accelerator_ids = runtime_ctx.get_accelerator_ids().get("GPU", [])
            if len(accelerator_ids) > 0:
                cuda_device = str(accelerator_ids[0])

        return {
            "host_ip": get_ip(host_ifname),
            "ray_node_id": str(runtime_ctx.get_node_id()),
            "ray_actor_id": str(runtime_ctx.get_actor_id()),
            "cuda_device": cuda_device,
        }
        
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
    
    def __init__(self):
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
    
    def create_request(self, req_id: int, max_output_len: int):
        self.req_max_output_len[req_id] = max_output_len
        self.output_len[req_id] = 0

    def clean_request(self, req_id: int):
        if req_id in self.req_max_output_len:
            del self.req_max_output_len[req_id]
        if req_id in self.output_len:
            del self.output_len[req_id]
                    
    def check_end(self, req_id) -> bool:
        return self.output_len[req_id] >= self.req_max_output_len[req_id]
