import importlib

ray = importlib.import_module("ray")
import torch
import os
import asyncio

PlacementGroupSchedulingStrategy = importlib.import_module(
    "ray.util.scheduling_strategies"
).PlacementGroupSchedulingStrategy

from disagmoe.frontend.ray_helper import init_cluster, get_global_placement_group, InitCoreArgs
from disagmoe.frontend.engine import Engine, EngineType
from disagmoe.frontend.datatypes import ChannelInfo, SloStat, TraceContext, SamplerStepInfo
from disagmoe.utils.placement import ModelPlacement, ColocatePlacement
from disagmoe.utils.utils import get_nccl_unique_id, Counter, StepInfo
from disagmoe.utils.metrics import Metric
from disagmoe.utils.logger import initialize_logger, get_logger
from disagmoe.utils.constants import *
from disagmoe.scheduler import get_dp_scheduler, DPScheduler
from disagmoe.config import CacheConfig, ModelConfig, EngineConfig
from disagmoe.env import ENV_VARS
_tokenizer_mod = importlib.import_module("disagmoe.frontend.tokenizer")
Tokenizer = _tokenizer_mod.Tokenizer
Detokenizer = _tokenizer_mod.Detokenizer

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
        assert self.slo_stat is not None
        return self.slo_stat
        
    async def put(self, slo_stat: SloStat):
        self.slo_stat = slo_stat
        async with self.finish_cond:
            self.finish_cond.notify()
        

class Controller:
    
    def __init__(
        self,
        n_node: int,
        n_gpu_per_node: int,
        host_ifname: str = "",
        nccl_ib_hca: str = "",
        nccl_ib_gid_index: str = "",
        expert_wise_schedule: bool = False,
        enable_nsys: bool = False,
    ):
        # NOTE(hogura|20241003): assigning n_worker of workers, each worker with 1 gpu
        self.n_worker = n_node * n_gpu_per_node
        self.n_gpu_per_node = n_gpu_per_node
        self.n_gpu_per_worker = 1
        self.n_cpu_per_worker = 3
        self.workers = []
        self.attn_workers = []
        self.device_ids = []
        self._profile_enabled = False
        self.req_id_generator = Counter(start=1)
        self.in_flight_reqs = set()
        self.end_flag = False
        self.request_results: Dict[int, AsyncResult] = dict()
        self.is_polling = False
        self.enable_nsys = enable_nsys
        self.expert_wise_schedule = expert_wise_schedule
        self.host_ifname = host_ifname
        self.nccl_ib_hca = nccl_ib_hca
        self.nccl_ib_gid_index = nccl_ib_gid_index
        
        self.dp_scheduler: Optional[DPScheduler] = None
        
        initialize_logger("controller")
        init_cluster(self.n_worker, self.n_cpu_per_worker, self.n_gpu_per_worker)
        self._create_engines()
        
    def init_tokenizer(self):
        self.tokenizer = Tokenizer.remote(attn_dp_size=self.model_config.dp_size)
        
        tokenizer_ports = [TOKENIZER_PORT_BASE + i for i in range(self.model_config.dp_size)]
        self.tokenizer_addrs = ray.get(self.tokenizer.init_tokenizer_sockets.remote(tokenizer_ports, self.host_ifname))
        
        self.detokenizer = Detokenizer.remote()
        detokenizer_port = DETOKENIZER_PORT_BASE
        self.detokenizer_addr = ray.get(self.detokenizer.init_detokenizer_socket.remote(detokenizer_port, self.host_ifname))
            
    def _create_engines(self):
        pg = get_global_placement_group()
        device_count = {}
        node_ids = {}
        
        for bundle_id, bundle in enumerate(pg.bundle_specs):
            n_cpus, n_gpus = int(bundle.get("CPU")), int(bundle.get("GPU"))
            
            ray_scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker_env_vars = dict(ENV_VARS)
            # worker_env_vars["NCCL_MAX_NCHANNELS"] = "1"
            if self.host_ifname:
                worker_env_vars["NCCL_SOCKET_IFNAME"] = self.host_ifname
            if self.nccl_ib_hca:
                worker_env_vars["NCCL_IB_HCA"] = self.nccl_ib_hca
            if self.nccl_ib_gid_index:
                worker_env_vars["NCCL_IB_GID_INDEX"] = self.nccl_ib_gid_index
            workers_env: Dict[str, object] = {
                "env_vars": worker_env_vars,
            }
            
            if self.enable_nsys:
                workers_env["nsight"] = "default"
            
            worker = ray.remote(
                num_cpus=n_cpus,
                num_gpus=n_gpus,
                scheduling_strategy=ray_scheduling_strategy,
                runtime_env=workers_env,
            )(Engine).remote()
            
            worker_ip = ray.get(worker.get_node_ip.remote(self.host_ifname))
            cur_device_on_worker = device_count.get(worker_ip, 0)
            device_count[worker_ip] = cur_device_on_worker + 1
            if worker_ip not in node_ids:
                node_ids[worker_ip] = len(node_ids)
            node_id = node_ids[worker_ip]
            
            device_id = node_id * self.n_gpu_per_node + cur_device_on_worker
            worker.set_device_id.remote(device_id)
            
            self.workers.append(worker)
            self.device_ids.append(device_id)
            
        get_logger().info(f"workers: {len(self.workers), self.device_ids, node_ids}")
    
    @property
    def all_workers(self):
        return self.workers
    
    @property
    def all_device_ids(self):
        return self.device_ids

    # For heterogeneous cluster
    def get_worker_identities(self) -> List[Dict[str, Union[str, int]]]:
        identities = ray.get([worker.get_worker_identity.remote(self.host_ifname) for worker in self.workers])
        result: List[Dict[str, Union[str, int]]] = []
        for identity, device_id in zip(identities, self.device_ids):
            item = dict(identity)
            item["device_id"] = int(device_id)
            result.append(item)
        return result
    
    def get_pairwise_nccl_ids(
            self, model_place: ModelPlacement
        ) -> Tuple[Dict[int, Dict[int, str]], 
                   Dict[int, Dict[int, str]]]:
        in_nccl_ids = {i: {} for i in model_place.in_device_ids.keys()}
        out_nccl_ids = {i: {} for i in model_place.out_device_ids.keys()}
        for i, js in model_place.out_device_ids.items():
            for j in js:
                uid = get_nccl_unique_id()
                in_nccl_ids[j][i] = uid
                out_nccl_ids[i][j] = uid
        return in_nccl_ids, out_nccl_ids
    
    def init_engine(
        self, 
        transport_name: str,
        model_place: ModelPlacement, 
        model_config: ModelConfig,
        engine_config: EngineConfig,
        cache_config: CacheConfig,
        attn_dp_weights: Optional[Dict[int, float]] = None,
        per_device_config: Optional[Dict[int, dict]] = None,
        gate_profile_file: Optional[str] = None,
        dp_policy: Optional[str] = None
    ):
        get_logger().debug(f"Initializing engine with model placement: {model_place}")
        
        self.model_config = model_config
        self.cache_config = cache_config
        
        self.init_tokenizer()
        
        # collect attention workers for kv-cache management
        for worker, device_id in zip(self.workers, self.device_ids):
            if len(model_place.attn_layer_ids_at(device_id)) > 0:
                self.attn_workers.append(worker)
        
        # broadcast the host ips of all devices
        device_2_host = {
            device_id: ray.get(worker.get_node_ip.remote(self.host_ifname))
                for worker, device_id in zip(self.all_workers, self.all_device_ids)
        }
        self.device_2_host = device_2_host
        get_logger().info(f"device_id to host_ip: {device_2_host}")
        ray.get([
            worker.set_hosts.remote(device_2_host)
                for worker in self.all_workers
        ])
        
        def determine_worker_type(device_id: int) -> EngineType:
            if model_place.is_hybrid:
                return EngineType.HYBRID
            return EngineType.ATTENTION if model_place.has_attn(device_id) else EngineType.EXPERT

        def get_worker_engine_config(device_id: int) -> EngineConfig:
            if per_device_config is not None and device_id in per_device_config:
                from copy import copy
                cfg = copy(engine_config)
                overrides = per_device_config[device_id]
                for key, val in overrides.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, val)
                return cfg
            return engine_config
        
        # setup attention & expert
        tasks = []
        for worker, device_id in zip(self.workers, self.device_ids):
            worker_type = determine_worker_type(device_id)
            rank = model_place.rank_at(device_id, num_expert_per_rank=model_config.num_experts_per_rank)
            if worker_type == EngineType.HYBRID or worker_type == EngineType.ATTENTION:
                assert rank is not None
                tokenizer_addr = self.tokenizer_addrs[rank]
                detokenizer_addr = self.detokenizer_addr
            else:
                tokenizer_addr = None
                detokenizer_addr = None
            tasks.append(
                worker.setup_engine.remote(
                    worker_type,
                    model_config=model_config,
                    engine_config=get_worker_engine_config(device_id),
                    cache_config=cache_config,
                    rank=rank,
                    tokenizer_addr=tokenizer_addr,
                    detokenizer_addr=detokenizer_addr,
                )
            )
        ray.get(tasks)
        
        # Optionally upload and broadcast gate profile bytes to workers.
        if gate_profile_file is not None and len(gate_profile_file) > 0:
            with open(gate_profile_file, "rb") as f:
                _gate_profile_bytes = f.read()
            data_ref = ray.put(_gate_profile_bytes)
            # send only to attention-capable workers
            ray.get([
                worker.load_gate_profile_bytes.remote(data_ref)
                    for worker, device_id in zip(self.workers, self.device_ids)
                    if model_place.has_attn(device_id)
            ])
            get_logger().info(f"Uploaded gate profile and broadcast to attention workers: {len(_gate_profile_bytes)} bytes")
        
        print(f"transport_name: {transport_name}")
        ray.get([w.set_transport.remote(transport_name) for w in self.all_workers])
        
        # All ranks should use the same nccl comm id
        inbound_nccl_ids, outbound_nccl_ids = self.get_pairwise_nccl_ids(model_place)
        
        # init core
        tasks = [
            worker.init_core.remote(
                InitCoreArgs(
                    world_size=len(self.workers),
                    layer_ids=model_place.layer_ids_at(device_id),
                    in_device_ids=model_place.in_device_ids_at(device_id),
                    out_device_ids=model_place.out_device_ids.get(device_id, []),
                    inbound_nccl_ids=inbound_nccl_ids.get(device_id, {}),
                    outbound_nccl_ids=outbound_nccl_ids.get(device_id, {}),
                    out_channel_infos=[
                        ChannelInfo(
                            model_place.expert_ids_at(out),
                            model_place.attn_layer_ids_at(out),
                            model_place.attn_dp_rank_at(out),
                        ) for out in model_place.out_device_ids.get(device_id, [])
                    ],
                    out_device_group_ids={
                        j: [device_id] + model_place.device_groups.get(j, [])
                            for j in model_place.out_device_ids.get(device_id, [])
                    },
                    device_group_ids=model_place.device_groups.get(device_id, []),
                    expert_ranks=model_place.out_expert_ranks_at(device_id),
                    local_attn_dp_rank=model_place.attn_dp_rank_at(device_id),
                    expert_wise_schedule=self.expert_wise_schedule,
                    local_num_experts=model_place.local_num_experts_at(device_id),
                    local_expert_ids=model_place.unique_expert_ids_at(device_id),
                )
            ) for worker, device_id in zip(self.workers, self.device_ids)
        ]
        ray.get(tasks)
        get_logger().info("Launched all workers successfully")
        
        if attn_dp_weights is not None:
            dp_weights = [attn_dp_weights.get(i, 1.0) for i in range(model_config.dp_size)]
            self.dp_scheduler = get_dp_scheduler(model_config.dp_size, cache_config.block_size, "weighted", weights=dp_weights)
        else:
            policy = dp_policy or "max"
            self.dp_scheduler = get_dp_scheduler(model_config.dp_size, cache_config.block_size, policy)
        
        self.model_place: ModelPlacement = model_place
        
    def release_kv_cache(self, req_ids: Union[int, List[int]]):
        if not isinstance(req_ids, list):
            req_ids = [req_ids]
        tasks = [worker.release_seqs.remote(req_ids) for worker in self.attn_workers]
        ray.get(tasks)
        
    def start_engine(self, non_blocking: bool = False):
        tasks = [worker.start.remote() for worker in self.workers]
        if not non_blocking:
            ray.get(tasks)
            print(f"all workers started")
        
    def get_new_req_id(self) -> int:
        req_id = next(self.req_id_generator)
        self.in_flight_reqs.add(req_id)
        return req_id
    
    async def process_finished_results(self, results: List[SloStat]):
        dp_scheduler = self.dp_scheduler
        assert dp_scheduler is not None
        finished_req_ids = [r.req_id for r in results]
        for req_id in finished_req_ids:
            self.in_flight_reqs.remove(req_id)
            dp_scheduler.del_seq(req_id)
        
        # deal with request results
        for result in results:
            await self.request_results[result.req_id].put(result)
            self.request_results.pop(result.req_id)

    def fetch_step_stats(self) -> List[Tuple[List[StepInfo], Dict[int, List[TraceContext]], Metric]]:
        return ray.get([worker.fetch_step_stats.remote() for worker in self.workers])

    def dump_advanced_logs(self, suffix: str = "", output_dir: str = "./advanced_logs"):
        """Tell each worker to dump its advanced logs to disk locally, then
        SSH-gather the files to the head node's output_dir."""
        import subprocess

        local_paths = ray.get([
            worker.dump_advanced_logs.remote(suffix) for worker in self.workers
        ])

        os.makedirs(output_dir, exist_ok=True)
        out_paths = []
        for device_id, remote_path in zip(self.device_ids, local_paths):
            if remote_path is None:
                continue
            host_ip = self.device_2_host[int(device_id)]
            dev_dir = os.path.join(output_dir, f"device_{device_id}")
            os.makedirs(dev_dir, exist_ok=True)
            try:
                subprocess.run(
                    ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                     "-rq", f"{host_ip}:{remote_path}/.", dev_dir],
                    timeout=60, check=False,
                )
            except Exception:
                pass
            out_paths.append(dev_dir)
        return out_paths
        
    def fetch_queueing_delays(self) -> Tuple[List[List[float]], List[List[float]]]:
        results = ray.get([worker.fetch_queueing_delays.remote() for worker in self.workers])
        attn = []
        exp = []
        for worker_id, result in enumerate(results):
            if self.model_place.has_attn(self.device_ids[worker_id]):
                attn.append(result)
            else:
                exp.append(result)
        return attn, exp
        
    def fetch_sampler_step_infos(self) -> List[SamplerStepInfo]:
        return ray.get(self.detokenizer.fetch_sampler_step_infos.remote())
        
    async def poll_finished_results(self) -> None:
        print(f"master start polling request")
        while self.is_polling:
            results = ray.get(self.detokenizer.fetch_finished_results.remote())
            if len(results) != 0:
                asyncio.create_task(self.process_finished_results(results))
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
    
    def start_polling_results(self):
        self.is_polling = True
        self._polling_task = asyncio.create_task(self.poll_finished_results())

    async def stop_polling_results(self):
        self.is_polling = False
        if hasattr(self, '_polling_task') and self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
            
    def put_single_request(self, input_len: int, output_len: int) -> AsyncResult:
        dp_scheduler = self.dp_scheduler
        assert dp_scheduler is not None
        req_id = self.get_new_req_id()
        res = AsyncResult(req_id)
        self.request_results[req_id] = res
        dp_scheduler.put_request(
            self.tokenizer.put_single_request.remote,
            req_id, input_len + output_len, input_len, output_len
        )
        return res
    
    def get_pool_snapshot(self) -> List[List[int]]:
        return ray.get([worker.get_pool_snapshot.remote() for worker in self.workers])
    
    def get_topk_pool_snapshot(self) -> List[List[int]]:
        return ray.get([worker.get_topk_pool_snapshot.remote() for worker in self.workers])
        
    def fetch_submitted_time(self) -> Dict[int, int]:
        return ray.get(self.tokenizer.fetch_submitted_time.remote())
    
    # def set_schedule_policy(self, policy: str):
    #     ray.get([
    #         worker.set_schedule_policy.remote(policy) for worker in self.workers
    #     ])
    
    # def set_schedule_block(self, step: int):
    #     ray.get([
    #         worker.set_schedule_block.remote(step) for worker in self.workers
    #     ])
    
    def reset_workers(self):
        tasks = [worker.reset.remote() for worker in self.workers] + [self.detokenizer.reset.remote()]
        ray.get(tasks)
    
    def stop_workers(self):
        self.end_flag = True
        tasks = [worker.terminate.remote() for worker in self.workers]
        ray.get(tasks)
        self.stop_profile()
        
    def init_profile(
        self, 
        profile_start_min_batch_size: int = 100, 
        profile_num_steps: int = 10, 
        profile_dir: Optional[str] = None
    ):
        if self._profile_enabled:
            return
        self._profile_enabled = True
        resolved_dir = None
        if profile_dir is not None:
            try:
                resolved_dir = os.path.abspath(profile_dir)
            except Exception:
                resolved_dir = profile_dir
        tasks = [worker.init_profile.remote(profile_start_min_batch_size, profile_num_steps, resolved_dir) for worker in self.workers]
        ray.get(tasks)
        
    def stop_profile(self):
        if not self._profile_enabled:
            return
        self._profile_enabled = False
        tasks = [worker.stop_profile.remote() for worker in self.workers]
        ray.get(tasks)
        
    async def start_scheduler(self):
        assert self.dp_scheduler is not None
        stats = {self.model_place.attn_dp_rank_at(device_id): 1 << 31 for device_id in self.device_ids if self.model_place.has_attn(device_id)}
        for worker, device_id in zip(self.workers, self.device_ids):
            if self.model_place.has_attn(device_id):
                rank = self.model_place.attn_dp_rank_at(device_id)
                stats[rank] = min(stats[rank], ray.get(worker.get_configured_kv_cache_blocks.remote()))
        self.dp_scheduler.start(stats)
    
    async def stop_scheduler(self):
        assert self.dp_scheduler is not None
        await self.dp_scheduler.terminate()

    def reset(self):
        self.in_flight_reqs.clear()
        self.request_results.clear()
        # self.req_id_generator.reset()
        self.reset_workers()

controller: Controller

def init_controller(
    n_node: int,
    n_gpu_per_node: int,
    host_ifname: str = "",
    nccl_ib_hca: str = "",
    nccl_ib_gid_index: str = "",
    expert_wise_schedule: bool = False,
    enable_nsys: bool = False,
):
    global controller
    controller = Controller(
        n_node,
        n_gpu_per_node,
        host_ifname=host_ifname,
        nccl_ib_hca=nccl_ib_hca,
        nccl_ib_gid_index=nccl_ib_gid_index,
        expert_wise_schedule=expert_wise_schedule,
        enable_nsys=enable_nsys,
    )
    return controller
