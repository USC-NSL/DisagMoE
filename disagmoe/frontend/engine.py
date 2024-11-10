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
from disagmoe.ops.memory import get_mappings_from_exp_ids, permute_tokens
from disagmoe.utils.logger import get_logger
from disagmoe.utils.utils import tensor_as_buf, get_ip, nvtx_range
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
        
        # !TODO(hogura|20241011): remove the frozen tensors
        self._frozen_tensors: List[torch.Tensor] = []
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
        # First append blocks for each seqeunce
        meta_py = AttentionBatchMetadata.from_c(meta_c)
        for i, seq_id in enumerate(meta_py.seq_ids[:meta_py.num_prefill_seqs]):
            if seq_id not in self.decode_seq_lens:
                self.block_mgr.allocate(seq_id, meta_py.prefill_seq_len[i])
            else:
                self.block_mgr.append_tokens(seq_id, meta_py.prefill_seq_len[i] - meta_py.prefill_query_len[i], meta_py.prefill_query_len[i])
            w = meta_py.prefill_seq_len[i]
            self.decode_seq_lens[seq_id] = w
        
        for seq_id in meta_py.seq_ids[meta_py.num_prefill_seqs:]:
            assert seq_id in self.decode_seq_lens, f"seq {seq_id} should no be in decoding phase"
            decode_seq_len = self.decode_seq_lens.get(seq_id)
            self.block_mgr.append_tokens(seq_id, decode_seq_len, 1)
            self.decode_seq_lens[seq_id] = decode_seq_len + 1
        
        self._logger.debug(f"new decode_seq_lens: {self.decode_seq_lens}")
        
        tokens_in_batch = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        decode_seq_lens = [self.decode_seq_lens.get(i, 0) for i in meta_py.seq_ids[meta_py.num_prefill_seqs:]]
        
        assert self.block_mgr is not None and meta_py is not None
        block_table = self.block_mgr.prepare_block_table(meta_c)
        block_table_stride = len(block_table) // tokens_in_batch
        block_table_cuda = torch.Tensor(block_table, device="cpu").type(torch.int32).view(tokens_in_batch, -1).to("cuda", non_blocking=True)
        
        assert len(block_table) % tokens_in_batch == 0
        slot_mapping = np.empty(tokens_in_batch, dtype=np.int64)
        slot_idx = 0
        # prefill tokens
        st = time.time_ns()
        for i in range(meta_py.num_prefill_seqs):
            q_len = meta_py.prefill_query_len[i]
            seq_len = meta_py.prefill_seq_len[i]
            for idx in range(seq_len - q_len, seq_len):
                block_id, id_in_block = idx // BLOCK_SIZE, idx % BLOCK_SIZE
                slot_mapping[slot_idx] = block_table[i * block_table_stride + block_id] * BLOCK_SIZE + id_in_block
                slot_idx += 1
        # decode tokens
        for i in range(meta_py.num_prefill_tokens, tokens_in_batch):
            last_idx = self.decode_seq_lens[meta_py.seq_ids[i]] - 1
            block_id, id_in_block = last_idx // BLOCK_SIZE, last_idx % BLOCK_SIZE
            slot_mapping[slot_idx] = block_table[i * block_table_stride + block_id] * BLOCK_SIZE + id_in_block
            slot_idx += 1
        
        def make_seqlens(lens):
            if not lens:
                return None
            seqlen = [0]
            for l in lens:
                seqlen.append(seqlen[-1] + l)
            result = torch.IntTensor(seqlen).cuda()
            return result
        
        return FlashAttentionMetadata(
            meta_py.num_prefill_seqs,
            meta_py.num_prefill_tokens,
            meta_py.num_decode_tokens,
            torch.from_numpy(slot_mapping).cuda(),
            seq_lens=meta_py.prefill_seq_len + decode_seq_lens,
            seq_lens_tensor=torch.IntTensor(meta_py.prefill_seq_len + decode_seq_lens).cuda(),
            max_query_len=max(meta_py.prefill_query_len + [0]),
            max_prefill_seq_len=max(meta_py.prefill_seq_len + [0]),
            max_decode_seq_len=max(decode_seq_lens + [0]),
            query_start_loc=make_seqlens(meta_py.prefill_query_len),
            seq_start_loc=make_seqlens(meta_py.prefill_seq_len + decode_seq_lens),
            context_lens_tensor= \
                [seq_len - que_len for seq_len, que_len in \
                    zip(meta_py.prefill_seq_len, meta_py.prefill_query_len)] + \
                [seq_len - 1 for seq_len in decode_seq_lens],
            block_tables=block_table_cuda,
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
        positions = torch.zeros(tensor.shape[0], dtype=torch.long).cuda()
        
        if mocking:
            # if mocking is enabled, the meta_c is a python AttentionBatchMetadata class
            meta_c = meta_c.to_c()

        attn_meta = self._pack_flash_attn_metadata(meta_c)
        
        # TODO(hogura|20241015): only top-1 expert currently
        hiddens, expert_ids = self.executor.execute(meta_c.layer_id, positions, tensor, attn_meta)
        expert_ids = torch.randint(0, self.model_config.num_experts, (meta_c.shape[0], )) # FIXME: remove the dummy expert
        expert_ids = expert_ids.view((meta_c.shape[0],))
        exp_mappings, exp_cnt = get_mappings_from_exp_ids(expert_ids, self.model_config.num_experts)
        hiddens = permute_tokens(hiddens, exp_mappings)
        
        if not mocking:
            new_meta_c = meta_c.to_metadata()
            new_meta_c.update_exp_ids(expert_ids.tolist(), exp_mappings.tolist())
        else:
            new_meta_c = meta_c
        
        return hiddens, new_meta_c
    
    @nvtx_range("engine.process_batch_expert")
    def process_batch_expert(self, 
                             meta_c: Metadata, 
                             tensor: Tensor) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, ExpertsExecutor)
        
        expert_ids = torch.LongTensor(meta_c.exp_ids, device="cpu")
        exp_mappings, exp_cnt = get_mappings_from_exp_ids(expert_ids, self.model_config.num_experts)
        permuted_tensor = permute_tokens(tensor, exp_mappings)
        meta_c.permute_token_infos(exp_mappings.tolist())
        
        batch_sizes = meta_c.get_expert_batch_sizes(self.model_config.num_experts)
        batch_sizes = torch.LongTensor(
            [batch_sizes[i] for i in self.inner_exp_rank],
            device="cpu",   # NOTE(hogura|20241014): grouped_gemm requires batch_sizes to be on cpu
        )
        output = self.executor.execute(meta_c.layer_id, permuted_tensor, batch_sizes)
        # 2. permute tokens back to <prefill><decode> order
        new_mapping = meta_c.sort_by_prefill_order()
        output = permute_tokens(output, torch.LongTensor(new_mapping, device="cpu"))
        meta_c.update_exp_ids([], [])
        meta_c.step_layer()
        
        return output, meta_c

    @nvtx_range("Engine.post_process")
    def post_process(self, output: Tensor, meta: Metadata) -> None:
        self._frozen_tensors.append(output)  # TODO(hogura|20241014): use pybind11.ref_count to control the reference
        batch: TensorBatch = TensorBatch_C()
        batch.data = output.data_ptr()
        batch.metadata = meta
        self.dispatcher.put(batch, 0)

    def loop(self):
        self._logger.info("starting expert engine loop")
        while not self.end_flag:
            # self.scheduler.wait_for_new_requests()  # !NOTE(hogura|20241008): will block this process!
            batch_info = self.scheduler.schedule()
            if not batch_info.data:
                continue
            
            meta: Metadata = batch_info.metadata
            tensor = tensor_as_buf(batch_info.data, meta.shape)
            
            output, meta = self._process_batch(meta, tensor)
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
        self.tokenizer.put_request(x.data_ptr(), shape)
        self._frozen_tensors.append(x)
        
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