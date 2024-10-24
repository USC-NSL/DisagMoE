import ray
from ray.util.placement_group import placement_group, PlacementGroup

_placement_group: PlacementGroup = None

def init_cluster(n_worker: int = 1, n_gpu_per_worker: int = 4):
    ray.init("auto")
    pg = placement_group([
        {"GPU": n_gpu_per_worker, "CPU": 0} for i in range(n_worker)
    ] + [{"GPU": 0, "CPU": 1}] * 2, strategy="PACK")
    ray.get(pg.ready(), timeout=10)
    global _placement_group
    _placement_group = pg
    print("workers", n_worker, "gpus", n_gpu_per_worker)

def get_global_placement_group():
    global _placement_group
    if _placement_group is None:
        init_cluster()
    return _placement_group