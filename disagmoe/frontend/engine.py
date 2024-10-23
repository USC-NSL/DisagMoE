import torch

from disagmoe.executor.executor import (Executor, ExpertsExecutor, AttnExecutor,
                                        ModelConfig, CacheConfig)
from disagmoe.frontend.adapter import Scheduler, MuDispatcher, Sampler, Tokenizer, BlockManager
from disagmoe.frontend.datatypes import (Metadata, ChannelInfo, TensorBatch, 
                                         AttentionBatchMetadata, SloStat)
from disagmoe.utils.logger import get_logger
from disagmoe.utils.utils import tensor_as_buf, get_ip, nvtx_range
from disagmoe.utils.constants import *

from vllm.attention import AttentionMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from typing import Optional, List, Dict, Union, Callable, Tuple
from threading import Thread

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from disagmoe_c import (init_engine, start_engine, init_sampler, init_tokenizer,
                        ChannelInfo as ChannelInfo_C,
                        TensorBatch as TensorBatch_C,
                        BlockManager as BlockManager_C)

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
        self.is_attn = False
        
        if device_id is not None:
            self._logger = get_logger(f"engine{device_id}")
            
        self.loop_thread = Thread(target=self.loop)
        
        # !TODO(hogura|20241011): remove the frozen tensors
        self._frozen_tensors: List[torch.Tensor] = []
        self._process_batch: Callable
        
        self.block_mgr: BlockManager = None
        
        self.decode_seq_lens = {}

    def init_core(
            self,
            layer_ids: List[int],
            in_device_ids: List[int],
            out_device_ids: List[int],
            out_channel_infos: List[ChannelInfo],
            nccl_ids: Dict[int, int]
        ):
        """
        NOTE(hogura|20241003): When using ray, all the device_id called to CUDA should become 0
        """
        self._logger.info(f"launching core: {layer_ids, in_device_ids, out_device_ids, out_channel_infos}")
        self.scheduler, self.a_scheduler, self.dispatcher = init_engine(
            self.device_id,
            self.is_attn,
            layer_ids,
            in_device_ids,
            out_device_ids,
            [ChannelInfo_C(info.expert_ids, info.attn_layer_ids) 
                for info in out_channel_infos],
            nccl_ids
        )
        
    def start(self):
        start_engine(self.scheduler, self.a_scheduler, self.dispatcher)
        if self.scheduler == None:
            # NOTE(hogura|20241014): mocking the scheduler
            self.scheduler = self.a_scheduler
        self.loop_thread.start()

    def set_device_id(self, device_id: int):
        self.device_id = device_id
        self._logger = get_logger(f"engine{device_id}")
        
    def set_is_attn(self, is_attn: bool):
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda:0")
        
        # TODO(hogura|20241014): upgrade this function to `setup_engine`
        self.is_attn = is_attn
        model_config = ModelConfig(hidden_size=HIDDEN_SIZE,
                                    num_heads=16, 
                                    num_kv_heads=8, 
                                    num_experts=N_EXPERTS, 
                                    intermediate_size=INTERMEDIATE_SIZE,
                                    dtype=torch.bfloat16)
        cache_config = CacheConfig(BLOCK_SIZE, 0.8, 2, "auto")
        cache_config.num_gpu_blocks = 2 ** 10
        
        if is_attn:
            self.executor = AttnExecutor(model_config, cache_config)
            self._process_batch = self.process_batch_attn
            self.block_mgr = BlockManager_C(BLOCK_SIZE, NUM_BLOCKS, RESERVED_BLOCKS)
        else:
            self.executor = ExpertsExecutor(model_config)    
            self._process_batch = self.process_batch_expert

    @nvtx_range("engine.pack_flash_attn_metadata")
    def _pack_flash_attn_metadata(
            self,
            meta: AttentionBatchMetadata
        ) -> FlashAttentionMetadata:
        # First append blocks for each seqeunce
        for i, seq_id in enumerate(meta.seq_ids[:meta.num_prefill_seqs]):
            if seq_id not in self.decode_seq_lens:
                self.block_mgr.allocate(seq_id, meta.prefill_seq_len[i])
            else:
                self.block_mgr.append_tokens(seq_id, meta.prefill_seq_len[i] - meta.prefill_query_len[i], meta.prefill_query_len[i])
            self.decode_seq_lens[seq_id] = meta.prefill_seq_len[i]
        
        for seq_id in meta.seq_ids[meta.num_prefill_seqs:]:
            assert seq_id in self.decode_seq_lens, f"seq {seq_id} should no be in decoding phase"
            decode_seq_len = self.decode_seq_lens.get(seq_id)
            self.block_mgr.append_tokens(seq_id, decode_seq_len, 1)
            self.decode_seq_lens[seq_id] = decode_seq_len + 1
        
        self._logger.debug(f"new decode_seq_lens: {self.decode_seq_lens}")
        
        decode_seq_lens = [self.decode_seq_lens.get(i, 0) for i in meta.seq_ids[meta.num_prefill_seqs:]]
        
        assert self.block_mgr is not None and meta is not None
        block_table = self.scheduler.prepare_block_table(meta, self.block_mgr)
        
        tokens_in_batch = meta.num_decode_tokens + meta.num_prefill_tokens
        slot_mapping = torch.empty(tokens_in_batch, dtype=torch.long, device="cpu")
        slot_idx = 0
        # prefill tokens
        for i in range(meta.num_prefill_seqs):
            q_len = meta.prefill_query_len[i]
            seq_len = meta.prefill_seq_len[i]
            for idx in range(seq_len - q_len, seq_len):
                block_id, id_in_block = idx // BLOCK_SIZE, idx % BLOCK_SIZE
                slot_mapping[slot_idx] = block_table[i][block_id] * BLOCK_SIZE + id_in_block
                slot_idx += 1
                
        # decode tokens
        for i in range(meta.num_prefill_tokens, tokens_in_batch):
            last_idx = self.decode_seq_lens[meta.seq_ids[i]] - 1
            block_id, id_in_block = last_idx // BLOCK_SIZE, last_idx % BLOCK_SIZE
            slot_mapping[slot_idx] = block_table[i][block_id] * BLOCK_SIZE + id_in_block
            slot_idx += 1
            
        block_table = [torch.IntTensor(block_list).cuda() for block_list in block_table]
        block_table = pad_sequence(block_table, batch_first=True, padding_value=0)
        
        def make_seqlens(lens):
            if not lens:
                return None
            seqlen = [0]
            for l in lens:
                seqlen.append(seqlen[-1] + l)
            return torch.IntTensor(seqlen).cuda()
        
        return FlashAttentionMetadata(
            meta.num_prefill_seqs,
            meta.num_prefill_tokens,
            meta.num_decode_tokens,
            slot_mapping.cuda(),
            seq_lens=meta.prefill_seq_len + decode_seq_lens,
            seq_lens_tensor=torch.IntTensor(meta.prefill_seq_len + decode_seq_lens).cuda(),
            max_query_len=max(meta.prefill_query_len + [0]),
            max_prefill_seq_len=max(meta.prefill_seq_len + [0]),
            max_decode_seq_len=max(decode_seq_lens + [0]),
            query_start_loc=make_seqlens(meta.prefill_query_len),
            seq_start_loc=make_seqlens(meta.prefill_seq_len + decode_seq_lens),
            context_lens_tensor= \
                [seq_len - que_len for seq_len, que_len in \
                    zip(meta.prefill_seq_len, meta.prefill_query_len)] + \
                [seq_len - 1 for seq_len in decode_seq_lens],
            block_tables=block_table,
            use_cuda_graph=False,
        )

    @nvtx_range("engine.process_batch_attn")
    def process_batch_attn(self, 
                           meta: AttentionBatchMetadata, 
                           tensor: Tensor,
                           mocking: bool = False) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, AttnExecutor)
        
        self._logger.debug(f"process batch AttentionBatchMetadata: {meta}")
        
        # TODO(hogura|20241014): fill the real positions
        positions = torch.zeros(tensor.shape[0], dtype=torch.long).cuda()
        
        attn_meta = self._pack_flash_attn_metadata(meta)
        
        # TODO(hogura|20241015): only top-1 expert currently
        hiddens, expert_ids = self.executor.execute(positions, tensor, attn_meta)
        expert_ids = expert_ids.view((meta.shape[0],))
        
        # TODO(hogura|20241015): test & update the experts
        if not mocking:
            new_exp_ids = expert_ids.tolist()
            new_meta = meta.to_metadata()
            new_meta.update_exp_ids(new_exp_ids, True)
        else:
            new_meta = meta
        
        return hiddens, new_meta
    
    @nvtx_range("engine.process_batch_expert")
    def process_batch_expert(self, 
                             meta: Metadata, 
                             tensor: Tensor) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, ExpertsExecutor)
        
        batch_sizes = torch.LongTensor(
            meta.get_expert_batch_sizes(N_EXPERTS)
        ).cpu() # NOTE(hogura|20241014): grouped_gemm requires batch_sizes to be on cpu
        output = self.executor.execute(tensor, batch_sizes)
        
        meta.step_layer()
        meta.update_exp_ids([-1] * meta.shape[0], False)
        
        return output, meta

    @nvtx_range("Engine.post_process")
    def post_process(self, output: Tensor, meta: Metadata) -> None:
        self._frozen_tensors.append(output)  # TODO(hogura|20241014): use pybind11.ref_count to control the reference
        batch: TensorBatch = TensorBatch_C()
        batch.data = output.data_ptr()
        batch.metadata = meta
        self.dispatcher.put(batch)

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
                  nccl_ids: Dict[int, int]):
        self.sampler = init_sampler(
            self.device_id,
            in_device_ids,
            out_device_ids,
            [ChannelInfo_C(info.expert_ids, info.attn_layer_ids) 
                for info in out_channel_infos],
        )
        self._logger.info("inited sampler")
        
    def start(self):
        self.sampler.start()
        
    def wait_for_n_requests(self, n_request) -> Dict[int, SloStat]:
        result = self.sampler.get_slo_stats(n_request)
        while len(result) == 0:
            # NOTE(hogura|20241022): get_slo_stats will return len=0 until #request==n_reqquest
            result = self.sampler.get_slo_stats(n_request)
        new_res = {
            k: SloStat(
                stat.t_prefill,
                stat.t_decode,
                stat.t_tokens,
            ) for k, stat in result.items()
        }
        return new_res
        
class TokenizerEngine(Engine):
    
    def __init__(self):
        super().__init__(None, None, None, TOKENIZER_DEV_ID)
        self.tokenizer: Tokenizer = None
        self.hidden_size = HIDDEN_SIZE
        
    def put_request(self, tokens: List[int]):
        # TODO(hogura|20241008): only #prefill = 1 now
        assert len(tokens) == 1
        shape = (len(tokens), self.hidden_size)
        # TODO(hogura|20241008): add a py-tokenizer here
        x = torch.ones(size=shape).type(torch.bfloat16)
        self._logger.info("tokenizer put 1 request")
        self.tokenizer.put_request(x.data_ptr(), shape)
        self._frozen_tensors.append(x)
    
    def gen_n_request(self, n_request: int):
        for i in range(n_request):
            self.put_request([i])
    
    def set_tokenizer_config(self, hidden_size: int):
        self.hidden_size = hidden_size
    
    def init_core(self, 
                  layer_ids: List[int], 
                  in_device_ids: List[int], 
                  out_device_ids: List[int], 
                  out_channel_infos: List[ChannelInfo], 
                  nccl_ids: Dict[int, int]):
        self.tokenizer = init_tokenizer(
            self.device_id,
            out_device_ids,
            [ChannelInfo_C(info.expert_ids, info.attn_layer_ids) 
                for info in out_channel_infos],
        )
        self._logger.info("inited tokenizer")
    
    def start(self):
        self.tokenizer.start()