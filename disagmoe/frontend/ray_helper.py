import ray
from ray.util.placement_group import placement_group, PlacementGroup
from dataclasses import dataclass
from typing import List, Dict, Tuple
from disagmoe.frontend.datatypes import ChannelInfo
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

@dataclass
class InitCoreArgs:
    layer_ids: List[int]
    
    # P2P Channels
    in_device_ids: List[int]
    out_device_ids: List[int]
    out_channel_infos: List[ChannelInfo]
    
    in_nccl_ids: Dict[int, int]
    out_nccl_ids: Dict[int, int]
    
    in_nccl_ids_ext: Dict[int, int]
    out_nccl_ids_ext: Dict[int, int]
    
    expert_ranks: List[Tuple[int, int, int]]
    expert_wise_schedule: bool = False
    
    # Group Channels
    out_device_group_ids: Dict[int, List[int]] = None
    device_group_ids: List[int] = None
    group_nccl_ids: Tuple[str, str, str] = ("", "", "")
    local_attn_dp_rank: int = 0