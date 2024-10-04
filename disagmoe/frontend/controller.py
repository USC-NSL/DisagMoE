import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from disagmoe.frontend.ray_helper import init_cluster, get_global_placement_group
from disagmoe.frontend.engine import Engine
from disagmoe.utils.placement import ModelPlacement

from disagmoe_c import get_nccl_unique_id, ChannelInfo

class Controller:
    
    def __init__(self, n_node: int, n_gpu_per_node: int):
        # NOTE(hogura|20241003): assigning n_worker of workers, each worker with 1 gpu, i.e. no TP yet.
        self.n_worker = n_node * n_gpu_per_node
        self.n_gpu_per_node = n_gpu_per_node
        self.n_gpu_per_worker = 1
        self.workers = []
        self.device_ids = []
        
        init_cluster(self.n_worker, self.n_gpu_per_worker)
        self._create_engines()
        
    def _create_engines(self):
        pg = get_global_placement_group()
        device_count = {}
        node_ids = {}
        
        for bundle_id, bundle in enumerate(pg.bundle_specs):
            if not bundle.get("GPU", 0):
                # TODO(hogura|20241003): sampler/tokenizer worker
                continue
            
            ray_scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=self.n_gpu_per_worker,
                scheduling_strategy=ray_scheduling_strategy,
            )(Engine).remote()
            
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
    
    def _get_nccl_ids(self, model_place: ModelPlacement):
        prs = {}
        nccl_ids = {}
        for i, js in model_place.out_device_ids.items():
            nccl_ids[i] = {}
            for j in js:
                p = sorted([i, j])
                if p not in prs:
                    prs[p] = get_nccl_unique_id()
                nccl_ids[i][j] = prs[p]
        return nccl_ids
    
    def init_engine(self, model_place: ModelPlacement):
        nccl_ids = self._get_nccl_ids(model_place)
        ray.get([
            worker.init_core.remote(
                layer_ids=model_place.layer_ids_at(device_id),
                in_device_ids=model_place.in_device_ids.get(device_id, []),
                out_device_ids=model_place.out_device_ids.get(device_id, []),
                out_channel_infos=ChannelInfo(
                    model_place.expert_ids_at(device_id),
                    model_place.attn_ids_at(device_id)
                ),
                nccl_ids=nccl_ids[device_id],
            )
                for worker, device_id in zip(self.workers, self.device_ids)
        ])
        
    def start_engine(self):
        ray.get([worker.start.remote() for worker in self.workers])

controller: Controller


def init_controller(n_node: int, n_gpu_per_node: int):
    global controller
    controller = Controller(n_node, n_gpu_per_node)
    return controller