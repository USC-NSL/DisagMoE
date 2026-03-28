import os
import ray
from ray.util.placement_group import placement_group, PlacementGroup
from dataclasses import dataclass
from typing import List, Dict, Tuple
from disagmoe.frontend.datatypes import ChannelInfo

_placement_group: PlacementGroup = None

def init_cluster(n_worker, n_cpu_per_worker=3, n_gpu_per_worker=1):
    if not ray.is_initialized():
        try:
            tmpdir_path = os.environ.get("RAY_TMPDIR", "/tmp/ray")
            ray.init(address="auto", _temp_dir=tmpdir_path)
        except ConnectionError:
            print("ray not initialized, now initializing a default ray cluster")
            ray.init()
        
    pg = placement_group([{"GPU": n_gpu_per_worker, "CPU": n_cpu_per_worker} for i in range(n_worker)], strategy="PACK")
    ray.get(pg.ready(), timeout=20)
    global _placement_group
    _placement_group = pg

def get_global_placement_group():
    global _placement_group
    return _placement_group

@dataclass
class InitCoreArgs:
    world_size: int
    layer_ids: List[int]
    
    # P2P Channels
    in_device_ids: List[int]
    out_device_ids: List[int]
    out_channel_infos: List[ChannelInfo]
    
    inbound_nccl_ids:  Dict[int, str]
    outbound_nccl_ids: Dict[int, str]
    
    expert_ranks: List[Tuple[int, int, int]]
    expert_wise_schedule: bool = False
    
    local_num_experts: int = 0
    local_expert_ids: List[int] = None
    
    # Group Channels
    out_device_group_ids: Dict[int, List[int]] = None
    device_group_ids: List[int] = None
    local_attn_dp_rank: int = 0
