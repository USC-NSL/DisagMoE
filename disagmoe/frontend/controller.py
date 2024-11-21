import ray
import ray.runtime_env
import torch
import os

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from disagmoe.frontend.ray_helper import init_cluster, get_global_placement_group
from disagmoe.frontend.engine import Engine, SamplerEngine, TokenizerEngine, EngineType
from disagmoe.frontend.datatypes import ChannelInfo, SloStat
from disagmoe.utils.placement import ModelPlacement
from disagmoe.utils.utils import get_nccl_unique_id
from disagmoe.utils.logger import get_logger
from disagmoe.utils.constants import *
from disagmoe.config import CacheConfig, ModelConfig
from disagmoe.env import ENV_VARS

from typing import List, Dict, Optional, Union, Tuple

class RequestIDGenerator:
        
    def __init__(self):
        self._req_id = 0
        
    def next(self) -> int:
        self._req_id += 1
        return self._req_id

class Controller:
    
    def __init__(self, n_node: int, n_gpu_per_node: int):
        # NOTE(hogura|20241003): assigning n_worker of workers, each worker with 1 gpu
        self.n_worker = n_node * n_gpu_per_node
        self.n_gpu_per_node = n_gpu_per_node
        self.n_gpu_per_worker = 1
        self.workers = []
        self.attn_workers = []
        self.device_ids = []
        self._logger = get_logger("controller")
        self.sampler_worker = None
        self.tokenizer_worker = None
        self.profile = False
        self.req_id_generator = RequestIDGenerator()
        self.in_flight_reqs = set()
        
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
                    "env_vars": ENV_VARS,
                    # "nsight": "default"
                },
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
    
    def _get_nccl_ids(
            self, model_place: ModelPlacement
        ) -> Tuple[Dict[int, Dict[int, str]], 
                   Dict[int, Dict[int, str]], 
                   Dict[Tuple[int], Tuple[str, str]]]:
        in_nccl_ids = {i: {} for i in model_place.in_device_ids.keys()}
        out_nccl_ids = {i: {} for i in model_place.out_device_ids.keys()}
        for i, js in model_place.out_device_ids.items():
            if i in [TOKENIZER_DEV_ID, SAMPLER_DEV_ID]:
                continue
            for j in js:
                if j in [TOKENIZER_DEV_ID, SAMPLER_DEV_ID]:
                    continue
                uid = get_nccl_unique_id()
                in_nccl_ids[j][i] = uid
                out_nccl_ids[i][j] = uid
        group_nccl_ids = {
            # NOTE(hogura|20241118): the first is for the channel in Pool, the second is for the channel in Scheduler
            # the third is for the allreduce in TP Group
            tuple(group): (get_nccl_unique_id(), get_nccl_unique_id(), get_nccl_unique_id())
                for group in model_place.device_groups.values()
        }
        # inter-group nccl ids, [expert -> TP group]
        for j, group in model_place.device_groups.items():
            if len(group) > 1 and j != group[0]: # is a worker
                root = group[0]
                in_nccl_ids[j] = in_nccl_ids[root]
        return in_nccl_ids, out_nccl_ids, group_nccl_ids
    
    def init_engine(self, 
                    model_place: ModelPlacement, 
                    model_config: Optional[ModelConfig] = None,
                    cache_config: Optional[CacheConfig] = None):
        if not model_config:
            # TODO: replace default model config
            model_config = ModelConfig(hidden_size=HIDDEN_SIZE,
                                        num_heads=16, 
                                        num_kv_heads=8, 
                                        num_experts=N_EXPERTS, 
                                        intermediate_size=INTERMEDIATE_SIZE,
                                        dtype=torch.bfloat16)
        if not cache_config:
            cache_config = CacheConfig(BLOCK_SIZE, 0.8, 2, "auto", 
                                       num_gpu_blocks=NUM_BLOCKS, 
                                       num_reserved_blocks=RESERVED_BLOCKS)
        in_nccl_ids, out_nccl_ids, group_nccl_ids = self._get_nccl_ids(model_place)
        
        # collect attention workers for kv-cache management
        for worker, device_id in zip(self.workers, self.device_ids):
            if len(model_place.attn_ids_at(device_id)) > 0:
                self.attn_workers.append(worker)
        
        ray.get([
            worker.setup_engine.remote(
                EngineType.ATTENTION if model_place.is_attn(device_id) else EngineType.EXPERT,
                model_config=model_config,
                cache_config=cache_config,
                rank=model_place.rank_at(device_id, num_expert_per_rank=model_config.num_experts_per_rank),
            )
                for worker, device_id in zip(self.workers, self.device_ids)
        ])
        ray.get([
            worker.setup_engine.remote(
                worker_type,
                model_config=model_config
            )
                for worker, worker_type in zip(
                    [self.tokenizer_worker, self.sampler_worker],
                    [EngineType.TOKENIZER, EngineType.SAMPLER]
                )
        ])
        tasks = [
            worker.init_core.remote(
                layer_ids=model_place.layer_ids_at(device_id),
                in_device_ids=model_place.in_device_ids_at(device_id, model_config.tp_enable_inter_group),
                out_device_ids=model_place.out_device_ids.get(device_id, []),
                out_channel_infos=[
                    ChannelInfo(
                        model_place.expert_ids_at(out),
                        model_place.attn_ids_at(out)
                    )
                        for out in model_place.out_device_ids.get(device_id, [])
                ],
                in_nccl_ids=in_nccl_ids.get(device_id, {}),
                out_device_group_ids={
                    j: [device_id] + model_place.device_groups.get(j, [])
                        for j in model_place.out_device_ids.get(device_id, [])
                },
                out_nccl_ids=out_nccl_ids.get(device_id, {}),
                device_group_ids=model_place.device_groups.get(device_id, []),
                group_nccl_ids=group_nccl_ids.get(
                    tuple(model_place.device_groups.get(device_id, [])), ("", "", "")),
            )
                for worker, device_id in zip(
                    self.workers + [self.sampler_worker, self.tokenizer_worker], 
                    self.device_ids + [SAMPLER_DEV_ID, TOKENIZER_DEV_ID]
                )
        ]
        self._logger.info("launched all tasks")
        ray.get(tasks)
        
    def release_kv_cache(self, req_ids: Union[int, List[int]]):
        if not isinstance(req_ids, list):
            req_ids = [req_ids]
        tasks = [worker.release_seqs.remote(req_ids) for worker in self.attn_workers]
        ray.get(tasks)
        
    def start_engine(self, non_blocking: bool = True):
        tasks = [worker.start.remote() for worker in self.workers + \
                    [self.sampler_worker, self.tokenizer_worker]]
        if not non_blocking:
            ray.get(tasks)
    
    def get_new_req_id(self) -> int:
        req_id = self.req_id_generator.next()
        self.in_flight_reqs.add(req_id)
        return req_id
            
    def put_single_request(self, input_len: List[int]):
        self.tokenizer_worker.put_single_request.remote(self.get_new_req_id(), input_len)
        
    def put_requests(self, input_lens: int):
        req_ids = [self.get_new_req_id() for _ in range(len(input_lens))]
        self.tokenizer_worker.put_requests.remote(req_ids, input_lens)
                
    def wait_for_requests(self, n_request: int) -> Dict[int, SloStat]:
        results = ray.get(self.sampler_worker.wait_for_n_requests.remote(n_request))
        # clean all in flight reqs as they are all done
        finished_req_ids = [req_id for req_id in self.in_flight_reqs]
        self.release_kv_cache(finished_req_ids)
        self.in_flight_reqs.clear()
        return results
    
    def stop_workers(self):
        self.stop_profile()
        tasks = [worker.terminate.remote() for worker in self.workers]
        ray.get(tasks)
        
    def start_profile(self):
        self.profile = True
        tasks = [worker.start_profile.remote() for worker in self.workers]
        ray.get(tasks)
        
    def stop_profile(self):
        if not self.profile:
            return
        tasks = [worker.stop_profile.remote() for worker in self.workers]
        ray.get(tasks)

controller: Controller


def init_controller(n_node: int, n_gpu_per_node: int):
    global controller
    controller = Controller(n_node, n_gpu_per_node)
    return controller