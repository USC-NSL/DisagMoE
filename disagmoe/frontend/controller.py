import ray
import ray.runtime_env
import torch
import os
import asyncio

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from disagmoe.frontend.ray_helper import init_cluster, get_global_placement_group, InitCoreArgs
from disagmoe.frontend.engine import Engine, SamplerEngine, TokenizerEngine, EngineType
from disagmoe.frontend.datatypes import ChannelInfo, SloStat, TraceContext
from disagmoe.utils.placement import ModelPlacement, ColocatePlacement
from disagmoe.utils.utils import get_nccl_unique_id, Counter, StepInfo
from disagmoe.utils.metrics import Metric
from disagmoe.utils.logger import get_logger
from disagmoe.utils.constants import *
from disagmoe.scheduler import get_dp_scheduler, DPScheduler
from disagmoe.config import CacheConfig, ModelConfig, SamplingConfig
from disagmoe.env import ENV_VARS

from asyncio import Future

from typing import List, Dict, Optional, Union, Tuple

class AsyncResult:
    
    def __init__(self, req_id: int):
        self.req_id = req_id
        self.finish_cond = asyncio.Condition()
        self.slo_stat = None
        
    async def wait(self):
        async with self.finish_cond:
            await self.finish_cond.wait()
        
    async def get(self) -> SloStat:
        await self.wait()
        return self.slo_stat
        
    async def put(self, slo_stat: SloStat):
        self.slo_stat = slo_stat
        async with self.finish_cond:
            self.finish_cond.notify()
        

class Controller:
    
    def __init__(self, n_node: int, n_gpu_per_node: int, enable_nsys=False):
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
        self._profile_enabled = False
        self.req_id_generator = Counter(start=1)
        self.in_flight_reqs = set()
        self.end_flag = False
        self.request_results: Dict[int, AsyncResult] = dict()
        self.is_polling = False
        self.enable_nsys = enable_nsys
        
        self.dp_scheduler: DPScheduler = None
        
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
                
            workers_env= {
                "env_vars": ENV_VARS,
            }
            if self.enable_nsys:
                workers_env["nsight"] = "default"
                
            worker = ray.remote(
                num_cpus=n_cpus,
                num_gpus=n_gpus,
                scheduling_strategy=ray_scheduling_strategy,
                runtime_env=workers_env,
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
        self._logger.info(f"workers: {len(self.workers), self.device_ids, node_ids}")
    
    @property
    def all_workers(self):
        return self.workers + [self.sampler_worker, self.tokenizer_worker]
    
    @property
    def all_device_ids(self):
        return self.device_ids + [SAMPLER_DEV_ID, TOKENIZER_DEV_ID]
    
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
                    cache_config: Optional[CacheConfig] = None,
                    sampling_config: Optional[SamplingConfig] = None):
        
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
            
        if not sampling_config:
            self.max_output_len = 64
            print(f"Sampler using default max output len: {self.max_output_len}")
        else:
            self.max_output_len = sampling_config.max_output_len
            
        in_nccl_ids, out_nccl_ids, group_nccl_ids = self._get_nccl_ids(model_place)
        in_nccl_ids_ext, out_nccl_ids_ext, group_nccl_ids_ext = self._get_nccl_ids(model_place)
        
        
        # collect attention workers for kv-cache management
        for worker, device_id in zip(self.workers, self.device_ids):
            if len(model_place.attn_layer_ids_at(device_id)) > 0:
                self.attn_workers.append(worker)
        
        # broadcast the host ips of all devices
        device_2_host = {
            device_id: ray.get(worker.get_node_ip.remote()) 
                for worker, device_id in zip(self.all_workers, self.all_device_ids)
        }
        self._logger.info(f"device_id to host_ip: {device_2_host}")
        ray.get([
            worker.set_hosts.remote(device_2_host)
                for worker in self.all_workers
        ])
        
        def determine_worker_type(device_id: int) -> EngineType:
            if device_id in [SAMPLER_DEV_ID, TOKENIZER_DEV_ID]:
                return EngineType.SAMPLER if device_id == SAMPLER_DEV_ID else EngineType.TOKENIZER
            if model_place.is_hybrid:
                return EngineType.HYBRID
            return EngineType.ATTENTION if model_place.has_attn(device_id) else EngineType.EXPERT
        
        # setup attention & expert
        ray.get([ 
            worker.setup_engine.remote(
                determine_worker_type(device_id),
                model_config=model_config,
                cache_config=cache_config,
                rank=model_place.rank_at(device_id, num_expert_per_rank=model_config.num_experts_per_rank),
            )
                for worker, device_id in zip(self.workers, self.device_ids)
        ])
        
        # setup tokenizer & sampler
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
        
        ray.get(self.sampler_worker.set_sampling_params.remote(self.max_output_len))
        
        # init core
        tasks = [
            worker.init_core.remote(
                InitCoreArgs(
                    layer_ids=model_place.layer_ids_at(device_id),
                    in_device_ids=model_place.in_device_ids_at(device_id, model_config.tp_enable_inter_group),
                    out_device_ids=model_place.out_device_ids.get(device_id, []),
                    out_channel_infos=[
                        ChannelInfo(
                            model_place.expert_ids_at(out),
                            model_place.attn_layer_ids_at(out),
                            model_place.attn_dp_rank_at(out),
                        )
                            for out in model_place.out_device_ids.get(device_id, [])
                    ],
                    in_nccl_ids=in_nccl_ids.get(device_id, {}),
                    out_nccl_ids=out_nccl_ids.get(device_id, {}),
                    in_nccl_ids_ext=in_nccl_ids_ext.get(device_id, {}),
                    out_nccl_ids_ext=out_nccl_ids_ext.get(device_id, {}),
                    out_device_group_ids={
                        j: [device_id] + model_place.device_groups.get(j, [])
                            for j in model_place.out_device_ids.get(device_id, [])
                    },
                    device_group_ids=model_place.device_groups.get(device_id, []),
                    group_nccl_ids=group_nccl_ids.get(
                        tuple(model_place.device_groups.get(device_id, [])), ("", "", "")),
                    expert_ranks=model_place.out_expert_ranks_at(device_id),
                    local_attn_dp_rank=model_place.attn_dp_rank_at(device_id),
                )
            ) for worker, device_id in zip(
                self.workers + [self.sampler_worker, self.tokenizer_worker], 
                self.device_ids + [SAMPLER_DEV_ID, TOKENIZER_DEV_ID]
            )
        ]
        ray.get(tasks)
        self._logger.info("launched all tasks")
        
        self.dp_scheduler = get_dp_scheduler(
            model_config.dp_size, self.max_output_len, cache_config.block_size, "max"
        )
        
        self.model_place: ModelPlacement = model_place
        
    def release_kv_cache(self, req_ids: Union[int, List[int]]):
        if not isinstance(req_ids, list):
            req_ids = [req_ids]
        tasks = [worker.release_seqs.remote(req_ids) for worker in self.attn_workers]
        ray.get(tasks)
        
    def start_engine(self, non_blocking: bool = False):
        tasks = [worker.start.remote() for worker in self.workers + \
                    [self.sampler_worker, self.tokenizer_worker]]
        if not non_blocking:
            ray.get(tasks)
            print(f"all workers started")
        
    
    def get_new_req_id(self) -> int:
        req_id = next(self.req_id_generator)
        self.in_flight_reqs.add(req_id)
        return req_id
    
    async def process_finished_results(self, results: List[SloStat]):
        # release request resources
        finished_req_ids = [r.req_id for r in results]
        self.release_kv_cache(finished_req_ids)
        for req_id in finished_req_ids:
            self.in_flight_reqs.remove(req_id)
            self.dp_scheduler.del_seq(req_id)
        
        # deal with request results
        for result in results:
            await self.request_results[result.req_id].put(result)
            self.request_results.pop(result.req_id)

    def fetch_step_stats(self) -> List[Tuple[List[StepInfo], Dict[int, List[TraceContext]], Metric]]:
        return ray.get([worker.fetch_step_stats.remote() for worker in self.workers])
        
    async def poll_finished_results(self) -> List[SloStat]:
        print(f"master start polling request")
        while not self.end_flag:
            results = ray.get(self.sampler_worker.fetch_finished_results.remote())
            if len(results) != 0:
                asyncio.create_task(self.process_finished_results(results))
            await asyncio.sleep(0)
    
    def start_polling_results(self):
        self.is_polling = True
        asyncio.create_task(self.poll_finished_results())
            
    def put_single_request(self, input_len: List[int]) -> AsyncResult:
        req_id = self.get_new_req_id()
        res = AsyncResult(req_id)
        self.request_results[req_id] = res
        self.dp_scheduler.put_request(
            self.tokenizer_worker.put_single_request.remote, req_id, input_len)
        return res
        
    def put_requests(self, input_lens: int) -> List[AsyncResult]:
        raise NotImplementedError("DP Scheduler not implemented here")
        req_ids = [self.get_new_req_id() for _ in range(len(input_lens))]
        results = [AsyncResult(req_id) for req_id in req_ids]
        for r in results:
            self.request_results[r.req_id] = r
        dp_ranks = self.dp_scheduler.schedule(req_ids)
        self.tokenizer_worker.put_requests.remote(req_ids, input_lens, dp_ranks)
        return results
        
    def wait_for_requests(self, n_request: int) -> Dict[int, SloStat]:
        results = ray.get(self.sampler_worker.wait_for_n_requests.remote(n_request))
        # clean all in flight reqs as they are all done
        finished_req_ids = [req_id for req_id in self.in_flight_reqs]
        self.release_kv_cache(finished_req_ids)
        self.in_flight_reqs.clear()
        return results
    
    def stop_workers(self):
        self.end_flag = True
        tasks = [worker.terminate.remote() for worker in self.workers]
        ray.get(tasks)
        self.stop_profile()
        
    def start_profile(self, profile_dir=None):
        self._profile_enabled = True
        tasks = [worker.start_profile.remote(profile_dir) for worker in self.workers]
        ray.get(tasks)
        
    def stop_profile(self):
        if not self._profile_enabled:
            return
        tasks = [worker.stop_profile.remote() for worker in self.workers]
        ray.get(tasks)
        
    async def start_scheduler(self):
        stats = {self.model_place.attn_dp_rank_at(device_id): 1 << 31 for device_id in self.device_ids if self.model_place.has_attn(device_id)}
        for worker, device_id in zip(self.workers, self.device_ids):
            if self.model_place.has_attn(device_id):
                rank = self.model_place.attn_dp_rank_at(device_id)
                stats[rank] = min(stats[rank], ray.get(worker.get_configured_kv_cache_blocks.remote()))
        self.dp_scheduler.start(stats)
    
    async def stop_scheduler(self):
        await self.dp_scheduler.terminate()

controller: Controller


def init_controller(n_node: int, n_gpu_per_node: int, enable_nsys=False):
    global controller
    controller = Controller(n_node, n_gpu_per_node, enable_nsys)
    return controller