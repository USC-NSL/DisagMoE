import torch

from disagmoe.executor.executor import (Executor, ExpertsExecutor, AttnExecutor,
                                        ModelConfig, CacheConfig)
from disagmoe.frontend.adapter import Scheduler, MuDispatcher, Sampler, Tokenizer, BlockManager
from disagmoe.frontend.datatypes import Metadata, ChannelInfo, TensorBatch, AttentionBatchMetadata
from disagmoe.utils.logger import get_logger
from disagmoe.utils.utils import tensor_as_buf, get_ip
from disagmoe.utils.constants import *

from vllm.attention import AttentionMetadata

from typing import Optional, List, Dict, Union, Callable, Tuple
from threading import Thread

from torch import Tensor

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
        torch.set_default_dtype(torch.bfloat16)
        
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
        # TODO(hogura|20241014): upgrade this function to `setup_engine`
        self.is_attn = is_attn
        model_config = ModelConfig(hidden_size=HIDDEN_SIZE,
                                    num_heads=16, 
                                    num_kv_heads=8, 
                                    num_experts=N_EXPERTS, 
                                    intermediate_size=INTERMEDIATE_SIZE,
                                    dtype=torch.bfloat16)
        cache_config = CacheConfig(32, 0.8, 2, "auto")
        cache_config.num_gpu_blocks = 2 ** 10
        
        if is_attn:
            self.executor = AttnExecutor(model_config, cache_config)
            self._process_batch = self.process_batch_attn
            self.block_mgr = BlockManager_C(BLOCK_SIZE, NUM_BLOCKS, RESERVED_BLOCKS)
        else:
            self.executor = ExpertsExecutor(model_config)    
            self._process_batch = self.process_batch_expert

    def process_batch_attn(self, 
                           meta: AttentionBatchMetadata, 
                           tensor: Tensor) -> Tuple[Tensor, Metadata]:
        assert isinstance(self.executor, AttnExecutor)
        
        # TODO(hogura|20241014): fill the real positions
        positions = torch.zeros([meta.shape[0]]).type(torch.bfloat16).cuda()
        
        slot_table_list = [self.block_mgr.get_slot_id(i) for i in meta.seq_ids]
        slot_table = torch.LongTensor(slot_table_list).cuda()
        
        attn_meta = AttentionMetadata(
            meta.num_prefill_seqs,
            meta.num_prefill_tokens,
            meta.num_decode_tokens,
            slot_table
        )
        
        # TODO(hogura|20241015): only top-1 expert currently
        hiddens, expert_ids = self.executor.execute(positions, tensor, attn_meta)
        
        new_exp_ids = expert_ids.tolist()
        
        # TODO(hogura|20241015): test & update the experts
        new_meta = meta.to_metadata()
        new_meta.update_exp_ids(new_exp_ids, True)
        
        return hiddens, new_meta
    
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
        x = torch.ones(size=shape).type(torch.float16)
        self._logger.info("tokenizer put 1 request")
        self.tokenizer.put_request(x.data_ptr(), shape)
        self._frozen_tensors.append(x)
    
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