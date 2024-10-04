import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from disagmoe.frontend.ray_helper import init_cluster, get_global_placement_group
from disagmoe.frontend.engine import Engine

class Controller:
    
    def __init__(self, n_node: int, n_gpu_per_node: int):
        # NOTE(hogura|20241003): assigning n_worker of workers, each worker with 1 gpu, i.e. no TP yet.
        self.n_worker = n_node * n_gpu_per_node
        self.n_gpu_per_node = n_gpu_per_node
        self.n_gpu_per_worker = 1
        
        init_cluster(self.n_worker, self.n_gpu_per_worker)
        self._init_engines()
        self.workers = []
        
    def _init_engines(self):
        pg = get_global_placement_group()
        device_count = {}
        node_ids = {}
        
        for bundle_id, bundle in enumerate(pg.bundle_specs):
            if not bundle.get("GPU", 0):
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
            
            worker.set_device_id(node_id * self.n_gpu_per_node + cur_device_on_worker)
            
            self.workers.append(worker)
        
        ray.get([worker.init_core.remote() for worker in self.workers])
        
    def start_engine(self):
        ray.get([worker.start.remote() for worker in self.workers])

controller: Controller


def init_controller(n_node: int, n_gpu_per_node: int):
    global controller
    controller = Controller(n_node, n_gpu_per_node)
    return controller