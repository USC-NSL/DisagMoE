from disagmoe.executor.executor import Executor, FFNExecutor, AttnExecutor
from disagmoe.frontend.adapter import Scheduler, Dispatcher
from disagmoe.frontend.datatypes import Metadata, ChannelInfo
from disagmoe.utils.logger import get_logger
from disagmoe.utils.utils import tensor_as_buf, get_ip

from typing import Optional, List, Dict
from threading import Thread

from disagmoe_c import init_engine, start_engine


class Engine:

    def __init__(self, 
                 is_attn: bool,
                 scheduler: Optional[Scheduler] = None, 
                 executor: Optional[Executor] = None, 
                 dispatcher: Optional[Dispatcher] = None, 
                 device_id: Optional[int] = None):
        self.is_attn = is_attn
        self.scheduler: Scheduler = scheduler
        self.executor: Executor = executor
        self.dispatcher: Dispatcher = dispatcher
        self.device_id = device_id
        self.end_flag = False
        
        if device_id is not None:
            self._logger = get_logger(f"engine{device_id}")
            
        self.loop_thread = Thread(target=self.loop)

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
        self.scheduler, self.dispatcher = init_engine(
            self.device_id,
            self.is_attn,
            layer_ids,
            in_device_ids,
            out_device_ids,
            out_channel_infos,
            nccl_ids
        )
        
    def start(self):
        start_engine(self.scheduler, self.dispatcher)
        self.loop_thread.start()

    def set_device_id(self, device_id: int):
        self.device_id = device_id
        self._logger = get_logger(f"engine{device_id}")

    def loop(self):
        while not self.end_flag:
            self.scheduler.wait_for_new_requests()
            batch_info = self.scheduler.schedule()
            
            meta: Metadata = batch_info.metadata
            tensor = tensor_as_buf(batch_info.data, meta.shape)
            
            output = self.executor.forward(tensor)
            if isinstance(self.executor, FFNExecutor):
                meta.step_layer()
                meta.update_exp_ids([-1] * meta.num_tokens, required_sort=False)
            else:
                # 1. gate func, dummy sleep
                # TODO
                # 2. permute memory layout & prepare metadata
                
                # TODO
                new_exp_ids = [0] * meta.num_tokens
                meta.update_exp_ids(new_exp_ids, required_sort=True)
            
            self.dispatcher.put(output.data_ptr(), meta)
    
    def terminate(self):
        self.end_flag = True
        
    def get_node_ip(self) -> str:
        return get_ip()
