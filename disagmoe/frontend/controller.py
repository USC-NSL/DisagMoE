import ray
import os

import ray.runtime_env
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from disagmoe.frontend.ray_helper import init_cluster, get_global_placement_group
from disagmoe.frontend.engine import Engine, SamplerEngine, TokenizerEngine
from disagmoe.frontend.datatypes import ChannelInfo
from disagmoe.utils.placement import ModelPlacement
from disagmoe.utils.utils import get_nccl_unique_id
from disagmoe.utils.logger import get_logger
from disagmoe.utils.constants import *

from typing import List, Dict

class Controller:
    
    def __init__(self, n_node: int, n_gpu_per_node: int):
        # NOTE(hogura|20241003): assigning n_worker of workers, each worker with 1 gpu, i.e. no TP yet.
        self.n_worker = n_node * n_gpu_per_node
        self.n_gpu_per_node = n_gpu_per_node
        self.n_gpu_per_worker = 1
        self.workers = []
        self.device_ids = []
        self._logger = get_logger("controller")
        self.sampler_worker = None
        self.tokenizer_worker = None
        
        init_cluster(self.n_worker, self.n_gpu_per_worker)
        self._create_engines()
        
    def _create_engines(self):
        pg = get_global_placement_group()
        device_count = {}
        node_ids = {}
        
        embedding_ids = [SAMPLER_DEV_ID, TOKENIZER_DEV_ID]
        
        for bundle_id, bundle in enumerate(pg.bundle_specs):
            if not bundle.get("GPU", 0):
                n_cpus, n_gpus = 1, 0
            else:
                n_cpus, n_gpus = 0, self.n_gpu_per_worker
            
            ray_scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            worker_cls = Engine
            if n_cpus != 0:
                worker_cls = TokenizerEngine if self.tokenizer_worker is None else SamplerEngine
            worker = ray.remote(
                num_cpus=n_cpus,
                num_gpus=n_gpus,
                scheduling_strategy=ray_scheduling_strategy,
                runtime_env={
                    # "nsight": "default"
                }
                # runtime_env={"env_vars": {k: v for k, v in os.environ.items()}},
            )(worker_cls).remote()
            
            if n_cpus != 0:
                if self.tokenizer_worker is None:
                    self.tokenizer_worker = worker
                    worker.set_device_id.remote(TOKENIZER_DEV_ID)
                else:
                    self.sampler_worker = worker
                    worker.set_device_id.remote(SAMPLER_DEV_ID)
                continue
            
            worker_ip = ray.get(worker.get_node_ip.remote())
            cur_device_on_worker = device_count.get(worker_ip, 0)
            device_count[worker_ip] = cur_device_on_worker + 1
            if worker_ip not in node_ids:
                node_ids[worker_ip] = len(node_ids)
            node_id = node_ids[worker_ip]
            
            device_id = node_id * self.n_gpu_per_node + cur_device_on_worker
            worker.set_device_id.remote(device_id)
            
            self.workers.append(worker)
            self.device_ids.append(device_id)
        print("#workers", len(self.workers))
    
    def _get_nccl_ids(self, model_place: ModelPlacement):
        prs = {}
        nccl_ids = {k: {} for k in model_place.out_device_ids}
        for i, js in model_place.out_device_ids.items():
            for j in js:
                p = tuple(sorted((i, j)))
                if p not in prs:
                    prs[p] = get_nccl_unique_id()
                nccl_ids[i][j] = prs[p]
                nccl_ids[j][i] = prs[p]
        # self._logger.info(f"nccl_ids {nccl_ids}")
        return nccl_ids
    
    def init_engine(self, model_place: ModelPlacement):
        nccl_ids = self._get_nccl_ids(model_place)
        ray.get([
            worker.set_is_attn.remote(
                len(model_place.attn_ids_at(device_id)) > 0
            )
                for worker, device_id in zip(self.workers, self.device_ids)
        ])
        tasks = [
            worker.init_core.remote(
                layer_ids=model_place.layer_ids_at(device_id),
                in_device_ids=model_place.in_device_ids.get(device_id, []),
                out_device_ids=model_place.out_device_ids.get(device_id, []),
                out_channel_infos=[
                    ChannelInfo(
                        model_place.expert_ids_at(out),
                        model_place.attn_ids_at(out)
                    )
                        for out in model_place.out_device_ids.get(device_id, [])
                ],
                nccl_ids=nccl_ids.get(device_id, {}),
            )
                for worker, device_id in zip(
                    self.workers + [self.sampler_worker, self.tokenizer_worker], 
                    self.device_ids + [SAMPLER_DEV_ID, TOKENIZER_DEV_ID]
                )
        ]
        self._logger.info("launched all tasks")
        ray.get(tasks)
        
    def start_engine(self, non_blocking: bool = True):
        tasks = [worker.start.remote() for worker in self.workers + [self.sampler_worker, self.tokenizer_worker]]
        if not non_blocking:
            ray.get(tasks)
            
    def put_request(self, tokens: List[int]):
        self.tokenizer_worker.put_request.remote(tokens)

controller: Controller


def init_controller(n_node: int, n_gpu_per_node: int):
    global controller
    controller = Controller(n_node, n_gpu_per_node)
    return controller