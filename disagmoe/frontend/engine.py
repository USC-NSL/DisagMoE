import torch
import time
import enum
import os

import numpy as np

from disagmoe.executor.executor import (Executor, ExpertsExecutor, AttnExecutor,
                                        ModelConfig, CacheConfig)
from disagmoe.frontend.adapter import Scheduler, MuDispatcher, Sampler, Tokenizer, BlockManager
from disagmoe.frontend.datatypes import (Metadata, ChannelInfo, TensorBatch, 
                                         AttentionBatchMetadata, SloStat)
from disagmoe.ops.memory import get_mappings_from_exp_ids, permute_tokens_cuda as permute_tokens
from disagmoe.utils.logger import get_logger
from disagmoe.utils.utils import get_ip, nvtx_range
from disagmoe.utils.constants import *
from disagmoe.utils.placement import ParallelConfig
from disagmoe.models.distributed import set_tensor_model_parallel_config, set_tensor_model_parallel_channel

from vllm.attention import AttentionMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from typing import Optional, List, Dict, Union, Callable, Tuple
from threading import Thread

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from disagmoe_c import (init_engine, start_engine, init_sampler, init_tokenizer,
                        ChannelInfo as ChannelInfo_C,
                        TensorBatch as TensorBatch_C,
                        BlockManager as BlockManager_C,
                        ParallelConfig as ParallelConfig_C)

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
        self.a_scheduler: Scheduler = scheduler
        self.executor: Executor = executor
        self.dispatcher: MuDispatcher = dispatcher
        self.device_id = device_id
        self.end_flag = False
        self.engine_type: EngineType = None
        self.model_config: ModelConfig = None
        
        if device_id is not None:
            self._logger = get_logger(f"engine{device_id}")
            
        self.loop_thread = Thread(target=self.loop)
        
        self._process_batch: Callable
        
        self.block_mgr: BlockManager = None
        
        self.decode_seq_lens = {}
        self.profiler = None
        self.inner_exp_rank = []
        
    def start_profile(self):
        assert self.device_id is not None, "Engine should be assigned with a device before profiling"
        
        if profile_dir := os.environ["DMOE_PROFILE_DIR"]:
            print(f"enable profiler, results stored at {profile_dir}")
        
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
        assert self.profiler is not None or \
            os.environ.get("DMOE_PROFILE_DIR", "") == ""
        if os.environ.get("DMOE_PROFILE_DIR", "") != "":
            self.profiler.stop()

    @property
    def is_attn(self):
        return self.engine_type == EngineType.ATTENTION

    def _is_attn(self):
        return self.is_attn

    def init_core(
            self,
            layer_ids: List[int],
            # P2P Channels
            in_device_ids: List[int],
            out_device_ids: List[int],
            out_channel_infos: List[ChannelInfo],
            nccl_ids: Dict[int, int],
            # Group Channels
            tensor_group_device_ids: List[int] = None,
            tensor_group_nccl_id: str = "",
            meta_group_device_ids: List[int] = None,
            meta_group_nccl_id: str = "",
        ):
        """
        NOTE(hogura|20241003): When using ray, all the device_id called to CUDA should become 0
        """
        self._logger.info(f"launching core: {layer_ids, in_device_ids, out_device_ids, out_channel_infos}")
        if meta_group_device_ids is None:
            meta_group_device_ids = []
        if tensor_group_device_ids is None:
            tensor_group_device_ids = []
        self.scheduler, self.a_scheduler, self.dispatcher = init_engine(
            self.device_id,
            self.is_attn,
            layer_ids,
            # P2P Channels
            in_device_ids,
            out_device_ids,
            [ChannelInfo_C(info.expert_ids, info.attn_layer_ids) 
                for info in out_channel_infos],
            nccl_ids,
            ParallelConfig_C(
                self.model_config.tp_size,
                self.model_config.ep_size,
                self.model_config.num_experts_per_rank,
            ),
            # Group Channels
            tensor_group_device_ids,
            tensor_group_nccl_id,
            meta_group_device_ids,
            meta_group_nccl_id,
        )
        set_tensor_model_parallel_channel(self.a_scheduler.get_channel() if self.a_scheduler is not None else None)
        
    def _switch_scheduler(self):
        if self.scheduler == None:
            self.scheduler = self.a_scheduler
        
    def start(self):
        start_engine(self.scheduler, self.a_scheduler, self.dispatcher)
        self._switch_scheduler()
        self.loop_thread.start()

    def set_device_id(self, device_id: int):
        self.device_id = device_id
        self._logger = get_logger(f"engine{device_id}")

    def setup_engine(self, 
                     engine_type: EngineType,
                     model_config: ModelConfig,
                     cache_config: CacheConfig = None,
                     rank: int = 0):
        torch.set_default_dtype(torch.bfloat16)
        if engine_type in [EngineType.ATTENTION, EngineType.EXPERT]:
            torch.set_default_device("cuda:0")
            stream = torch.cuda.Stream()
            torch.cuda.set_stream(stream)
            
        set_tensor_model_parallel_config(model_config)
        self.engine_type = engine_type
        self.model_config = model_config
        if engine_type == EngineType.ATTENTION:
            self.executor = AttnExecutor.build(model_config, cache_config)
            self._process_batch = self.process_batch_attn
            self.block_mgr = BlockManager_C(
                cache_config.block_size, 
                cache_config.num_gpu_blocks, 
                cache_config.num_reserved_blocks)
        elif engine_type == EngineType.EXPERT:
            self.executor = ExpertsExecutor(model_config)
            self._process_batch = self.process_batch_expert
            # prepare inner exp rank, [n_exp_per_rank * rank, (rank + 1) * n_exp_per_rank) -> [0, n_exp_per_rank)
            self.inner_exp_rank = [0 for _ in range(model_config.num_experts_per_rank)]
            for i in range(model_config.num_experts_per_rank):
                self.inner_exp_rank[i] = model_config.num_experts_per_rank * rank + i
        
        self._logger.info(f"engine setup.")

    @nvtx_range("engine.pack_flash_attn_metadata")
    def _pack_flash_attn_metadata(
            self,
            meta_c: AttentionBatchMetadata
        ) -> FlashAttentionMetadata:
        meta_py = AttentionBatchMetadata.from_c(meta_c)
        
        prefill_seq_ids = meta_py.seq_ids[:meta_py.num_prefill_seqs]
        decode_seq_ids = meta_py.seq_ids[meta_py.num_prefill_seqs:]
        
        decode_seq_lens = [self.decode_seq_lens.get(seq_id) for seq_id in decode_seq_ids]
        
        # 1. update and prepare block table
        self.block_mgr.update_block_table(meta_c, decode_seq_lens)
        block_table, slot_mapping = self.block_mgr.prepare_block_table(meta_c, decode_seq_lens)
        block_table_cuda = torch.IntTensor(block_table, device="cpu").to("cuda", non_blocking=True)
        slot_mapping_cuda = torch.LongTensor(slot_mapping, device="cpu").to("cuda", non_blocking=True)
        tokens_in_batch = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        assert len(block_table) % tokens_in_batch == 0
        
        # 2. update decode_seq_lens
        for i, seq_id in enumerate(prefill_seq_ids):
            self.decode_seq_lens[seq_id] = meta_py.prefill_seq_len[i]
        
        for i, seq_id in enumerate(decode_seq_ids):
            decode_seq_lens[i] += 1
            self.decode_seq_lens[seq_id] += 1
        
        self._logger.debug(f"new decode_seq_lens: {self.decode_seq_lens}")
        
        # 3. prepare seqlens and start_locs
        seq_lens_cuda = torch.IntTensor(meta_py.prefill_seq_len + decode_seq_lens).to("cuda", non_blocking=True)
        
        def make_seqlens(lens):
            if not lens:
                return None
            seqlen = [0]
            for l in lens:
                seqlen.append(seqlen[-1] + l)
            result = torch.IntTensor(seqlen).to("cuda", non_blocking=True)
            return result
        
        query_start_loc = make_seqlens(meta_py.prefill_query_len)
        seq_start_loc = make_seqlens(meta_py.prefill_seq_len + decode_seq_lens)
        context_lens_tensor = [seq_len - que_len for seq_len, que_len in \
                    zip(meta_py.prefill_seq_len, meta_py.prefill_query_len)] + \
                [seq_len - 1 for seq_len in decode_seq_lens]
        
        return FlashAttentionMetadata(
            meta_py.num_prefill_seqs,
            meta_py.num_prefill_tokens,
            meta_py.num_decode_tokens,
            slot_mapping_cuda,
            seq_lens=meta_py.prefill_seq_len + decode_seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=max(meta_py.prefill_query_len + [0]),
            max_prefill_seq_len=max(meta_py.prefill_seq_len + [0]),
            max_decode_seq_len=max(decode_seq_lens + [0]),
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_table_cuda.view(tokens_in_batch, -1),
            use_cuda_graph=False,
        )

    @nvtx_range("engine.process_batch_attn")
    def process_batch_attn(self, 
                           meta_c: AttentionBatchMetadata, 
                           tensor: Tensor,
                           mocking: bool = False) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, AttnExecutor)
        
        self._logger.debug(f"process batch AttentionBatchMetadata: {meta_c}")
        
        # TODO(hogura|20241014): fill the real positions
        positions = torch.zeros(tensor.shape[0], dtype=torch.long).to("cuda", non_blocking=True)
        
        if mocking:
            # if mocking is enabled, the meta_c is a python AttentionBatchMetadata class
            meta_c = meta_c.to_c()

        attn_meta = self._pack_flash_attn_metadata(meta_c)
        
        # TODO(hogura|20241015): only top-1 expert currently
        hiddens, expert_ids = self.executor.execute(meta_c.layer_id, positions, tensor, attn_meta)
        expert_ids = torch.randint(0, self.model_config.num_experts, (meta_c.shape[0], )) # FIXME: remove the dummy expert
        expert_ids = expert_ids.view((meta_c.shape[0],)).tolist()
        exp_mappings, _ = get_mappings_from_exp_ids(expert_ids, self.model_config.num_experts)
        hiddens = permute_tokens(hiddens, exp_mappings)
        
        if not mocking:
            new_meta_c = meta_c.to_metadata()
            new_meta_c.update_exp_ids(expert_ids, exp_mappings)
        else:
            new_meta_c = meta_c
        
        return hiddens, new_meta_c
    
    @nvtx_range("engine.process_batch_expert")
    def process_batch_expert(self, 
                             meta_c: Metadata, 
                             tensor: Tensor) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, ExpertsExecutor)
        
        exp_mappings, exp_cnt = get_mappings_from_exp_ids(meta_c.exp_ids, self.model_config.num_experts)
        permuted_tensor = permute_tokens(tensor, exp_mappings)
        meta_c.permute_token_infos(exp_mappings)
        
        # OPTIMIZE(shaoyuw): use exp_cnt to get batch_sizes
        batch_sizes = meta_c.get_expert_batch_sizes(self.model_config.num_experts)
        batch_sizes = torch.tensor(
            [batch_sizes[i] for i in self.inner_exp_rank],
            dtype=torch.int64,
            device="cpu",   # NOTE(hogura|20241014): grouped_gemm requires batch_sizes to be on cpu
        )
        output = self.executor.execute(meta_c.layer_id, permuted_tensor, batch_sizes)
        # 2. permute tokens back to <prefill><decode> order
        new_mappings = list(meta_c.sort_by_prefill_order())
        output = permute_tokens(output, new_mappings)
        meta_c.update_exp_ids([], [])
        meta_c.step_layer()
        
        return output, meta_c

    @nvtx_range("Engine.post_process")
    def post_process(self, output: Tensor, meta: Metadata) -> None:
        batch: TensorBatch = TensorBatch_C()
        batch.data = output
        batch.metadata = meta
        self.dispatcher.put(batch, 0)

    def loop(self):
        self._logger.info("starting expert engine loop")
        while not self.end_flag:
            # self.scheduler.wait_for_new_requests()  # !NOTE(hogura|20241008): will block this process!
            batch_info = self.scheduler.schedule()
            if batch_info.data is None:
                continue
            
            batch = TensorBatch.from_c(batch_info)

            meta: Metadata = batch.metadata
            output, meta = self._process_batch(meta, batch.data)
            self.post_process(output, meta)
    
    def terminate(self):
        self.end_flag = True
        
    def get_node_ip(self) -> str:
        return get_ip()

class SamplerEngine(Engine):
    
    def __init__(self):
        super().__init__(None, None, None, SAMPLER_DEV_ID)
        self.sampler: Sampler = None
        
    def init_core(self, 
                  layer_ids: List[int], 
                  in_device_ids: List[int], 
                  out_device_ids: List[int], 
                  out_channel_infos: List[ChannelInfo], 
                  nccl_ids: Dict[int, int],
                  tensor_group_device_ids: List[int] = None,
                  tensor_group_nccl_id: str = "",
                  meta_group_device_ids: List[int] = None,
                  meta_group_nccl_id: str = ""):
        self.sampler = init_sampler(
            self.device_id,
            in_device_ids,
            out_device_ids,
            [ChannelInfo_C(info.expert_ids, info.attn_layer_ids) 
                for info in out_channel_infos],
        )
        self._logger.info("inited sampler")
        self._t_start = time.time()
        
    def start(self):
        self.sampler.start()
        
    def wait_for_n_requests(self, n_request) -> Dict[int, SloStat]:
        result = self.sampler.get_slo_stats(n_request)
        while len(result) == 0:
            # NOTE(hogura|20241022): get_slo_stats will return len=0 until #request==n_reqquest
            result = self.sampler.get_slo_stats(n_request)
        new_res = {
            k: SloStat(
                stat.t_prefill / CPS - (time.time() - self._t_start),
                (stat.t_decode - stat.t_prefill) / CPS,
                [(x - y) / CPS for x, y in zip(stat.t_tokens[1:], stat.t_tokens[:-1])],
            ) for k, stat in result.items()
        }
        return new_res
        
class TokenizerEngine(Engine):
    
    def __init__(self):
        super().__init__(None, None, None, TOKENIZER_DEV_ID)
        self.tokenizer: Tokenizer = None
        
    def put_request(self, tokens: List[int]):
        # TODO(hogura|20241008): only #prefill = 1 now
        assert len(tokens) == 1
        shape = (len(tokens), self.model_config.hidden_size)
        # TODO(hogura|20241008): add a py-tokenizer here
        x = torch.ones(size=shape).type(self.model_config.dtype)
        self._logger.info("tokenizer put 1 request")
        self.tokenizer.put_request(x, shape)
        
    def gen_n_request(self, n_request: int):
        for i in range(n_request):
            self.put_request([i])
        
    def init_core(self, 
                  layer_ids: List[int], 
                  in_device_ids: List[int], 
                  out_device_ids: List[int], 
                  out_channel_infos: List[ChannelInfo], 
                  nccl_ids: Dict[int, int],
                  tensor_group_device_ids: List[int] = None,
                  tensor_group_nccl_id: str = "",
                  meta_group_device_ids: List[int] = None,
                  meta_group_nccl_id: str = ""):
        self.tokenizer = init_tokenizer(
            self.device_id,
            out_device_ids,
            [ChannelInfo_C(info.expert_ids, info.attn_layer_ids) 
                for info in out_channel_infos],
        )
        self._logger.info("inited tokenizer")
    
    def start(self):
        self.tokenizer.start()