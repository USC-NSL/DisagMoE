from disagmoe.utils.constants import *
from disagmoe.utils.utils import Counter
from disagmoe.config import ModelConfig
from disagmoe.utils.logger import get_logger

from typing import Dict, Tuple, Optional, Union, List, override
from dataclasses import dataclass

from itertools import product

@dataclass
class ParallelConfig:
    tp: int = 1
    ep: int = 1
    dp: int = 1
    n_exp_per_rank: int = 1
    expert_ranks: Dict[Tuple[int, int], int] = None
    
    @staticmethod
    def to_c(tp: int, ep: int, dp: int, n_exp_per_rank: int, expert_ranks: List, n_total_experts: int = 0) -> "ParallelConfig_C":
        from disagmoe_c import ParallelConfig as ParallelConfig_C
        cfg = ParallelConfig_C()
        cfg.tp = tp
        cfg.ep = ep
        cfg.dp = dp
        cfg.n_exp_per_rank = n_exp_per_rank
        cfg.n_total_experts = n_total_experts
        cfg.expert_ranks = expert_ranks
        return cfg

@dataclass
class ModelPlacement:
    # device_id -> layer_id
    attn: Dict[int, List[int]]
    
    # device_id -> list(layer_id, expert_id)
    expert: Dict[int, List[Tuple[int, int]]]
    
    # for the devices in a TP group, only the driver's device_id is stored in the edges
    in_device_ids: Dict[int, List[int]]
    out_device_ids: Dict[int, List[int]]
    
    device_groups: Dict[int, List[int]] = None
    
    # (layer_id, expert_id) -> expert_rank_id
    expert_ranks: Dict[Tuple[int, int], int] = None
    
    # device_id -> attn_rank_id
    attn_dp_ranks: Dict[int, int] = None
    
    # device_id -> number of unique experts on this device (asymmetric placement)
    local_expert_counts: Dict[int, int] = None
    
    is_hybrid: bool = False
        
    def expert_rank_at(self, device_id: int, num_expert_per_rank: int) -> int:
        assert device_id in self.expert
        assert self.expert_ranks is not None
        ids = self.expert[device_id]
        ranks = []
        for layer_id, expert_id in ids:
            ranks.append(self.expert_ranks[(layer_id, expert_id)])
        assert len(set(ranks)) == 1
        return ranks[0]
    
    def attn_rank_at(self, device_id: int) -> int:
        assert device_id in self.attn
        for i, d in enumerate(self.device_groups[device_id]):
            if d == device_id:
                return i
        return 0
    
    def rank_at(self, device_id: int, *args, **kwargs) -> int:
        if device_id in self.expert:
            return self.expert_rank_at(device_id, *args, **kwargs)
        else:
            return self.attn_rank_at(device_id)
    
    def out_expert_ranks_at(self, device_id: int) -> List[Tuple[int, int, int]]:
        if device_id not in self.out_device_ids:
            return []
        result = []
        for dev_out in self.out_device_ids[device_id]:
            if dev_out in self.expert:
                for layer_id, expert_id in self.expert[dev_out]:
                    result.append((layer_id, expert_id, self.expert_ranks[(layer_id, expert_id)]))
        return result
        
    def attn_layer_ids_at(self, device_id: int) -> List[int]:
        return self.attn.get(device_id, [])
    
    def expert_ids_at(self, device_id: int):
        return self.expert.get(device_id, [])
    
    def unique_expert_ids_at(self, device_id: int) -> List[int]:
        return sorted(set(eid for _, eid in self.expert.get(device_id, [])))
    
    def local_num_experts_at(self, device_id: int) -> int:
        if self.local_expert_counts is not None and device_id in self.local_expert_counts:
            return self.local_expert_counts[device_id]
        return len(self.unique_expert_ids_at(device_id))
    
    def attn_dp_rank_at(self, device_id: int) -> int:
        return self.attn_dp_ranks.get(device_id, 0)
    
    def layer_ids_at(self, device_id: int) -> List[int]:
        return sorted(list(set(
            self.attn.get(device_id, []) + [e[0] for e in self.expert.get(device_id, [])]
        )))
        
    def add_edge(self, src, dst):
        # assert src != dst
        if dst not in self.in_device_ids:
            self.in_device_ids[dst] = []
        if src not in self.out_device_ids:
            self.out_device_ids[src] = []
        if src not in self.in_device_ids[dst]:
            self.in_device_ids[dst].append(src)
            self.out_device_ids[src].append(dst)
            
    def is_worker_device(self, device_id: int) -> bool:
        return device_id in self.device_groups and self.device_groups[device_id][0] != device_id
    
    def has_attn(self, device_id: int) -> bool:
        return device_id in self.attn
    
    def in_device_ids_at(self, device_id: int) -> List[int]:
        return self.in_device_ids.get(device_id, [])

@dataclass
class ClusterConfig:
    n_node: int
    n_gpu: int
    gpu_cap: float = 40 * GiB

class PlacementBase:
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig, 
                 step_attn: int = 0, step_expert: int = 0, 
                 zigzag_attn: bool = True, expert_allocation: Optional[List[int]] = None):
        self.model_config = model_config
        self.cluster_config = cluster_config
        # For heterogeneous cluster
        self.expert_allocation = expert_allocation
        if self.expert_allocation is not None:
            assert sum(self.expert_allocation) == model_config.num_experts, \
                f"Sum of expert allocation {sum(self.expert_allocation)} must match num_experts {model_config.num_experts}"
        
    @property
    def tp_size(self):
        return self.model_config.tp_size
    
    @property
    def ep_size(self):
        return self.model_config.ep_size
    
    @property
    def dp_size(self):
        return self.model_config.dp_size
    
    @property
    def num_layers(self):
        return self.model_config.num_layers
        
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        raise NotImplementedError()
    
    def _add_edges(self, place: ModelPlacement) -> ModelPlacement:
        attn_devs = { layer_id: [] for layer_id in range(self.num_layers) }
        exp_devs = { layer_id: [] for layer_id in range(self.num_layers) }
        for dev, layer_ids in place.attn.items():
            for layer_id in layer_ids:
                attn_devs[layer_id].append(dev)
        for dev, layer_ids in place.expert.items():
            for layer_id, exp_id in layer_ids:
                exp_devs[layer_id].append(dev)
        
        for layer_id in range(self.num_layers):
            if layer_id > 0:
                # last exp to current attn
                for dev in attn_devs[layer_id]:
                    for prev_dev in exp_devs[layer_id - 1]:
                        place.add_edge(prev_dev, dev)
                        
            # current attn to current exp
            for dev in attn_devs[layer_id]:
                for exp_dev in exp_devs[layer_id]:
                    place.add_edge(dev, exp_dev)
        
        # connect first attn layer with last exp layer
        for dev in attn_devs[0]:
            for prev_dev in exp_devs[self.num_layers - 1]:
                place.add_edge(prev_dev, dev)
                    
        return place
        
    def _update_expert_rank(self, place: ModelPlacement) -> ModelPlacement:
        """
            default EP worker rank is `expert_id // num_experts_per_rank` for each expert
        """
        if self.expert_allocation is not None:
            expert_ranks = {}
            for dev_id, layers_experts in place.expert.items():
                for layer_id, expert_id in layers_experts:
                    expert_ranks[(layer_id, expert_id)] = dev_id
            place.expert_ranks = expert_ranks
        else:
            expert_ranks = {
                (layer_id, expert_id): expert_id // self.model_config.num_experts_per_rank
                    for layer_id, expert_id in product(range(self.num_layers),
                                                    range(self.model_config.num_experts))
            }
            place.expert_ranks = expert_ranks
        return place
    
    def _update_attn_dp_rank(self, place: ModelPlacement) -> ModelPlacement:
        """
            DP is not enabled in default strategy
        """
        attn_dp_ranks = {
            dev_id: 0
                for dev_id in place.attn
        }
        place.attn_dp_ranks = attn_dp_ranks
        return place
    
    def solve(self) -> ModelPlacement:
        place = self._solve(
            self.model_config.num_layers, self.model_config.num_experts,
            self.cluster_config.n_node, self.cluster_config.n_gpu
        )
        place = self._add_edges(place)
        place = self._update_expert_rank(place)
        place = self._update_attn_dp_rank(place)
        return place


class SinglePlacement(PlacementBase):
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig, rep_attn: int=1):
        super().__init__(model_config, cluster_config)
        self.rep_attn = rep_attn
    
    @override
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        # 1 attn, n_expert experts
        assert n_layer * (self.rep_attn + n_expert) <= n_node * n_gpu_per_node
        # not considering gpu_cap yet
        attn = {}
        expert = {}
        node_id = Counter()
        pg = ModelPlacement(
            attn, expert, {}, {}
        )
        for i in range(n_layer):
            attns = []
            for j in range(self.rep_attn):
                i_attn = next(node_id)
                attn[i_attn] = [i]
                attns.append(i_attn)
            
            i_last_experts = []
            for j in range(n_expert):
                i_expert = next(node_id)
                i_last_experts.append(i_expert)
                expert[i_expert] = [(i, j)]

        assert len(attn) == n_layer * self.rep_attn
        assert len(pg.in_device_ids) == n_layer * (self.rep_attn + n_expert) + 1
        assert len(pg.out_device_ids) == n_layer * (self.rep_attn + n_expert) + 1
        return pg

class InterleavePlacement(PlacementBase):
    
    """
    The structure of the placement is as follows:
    ```
    (Attn_0, Attn_{n_layer // n_group}, ...), (Expert_0, Expert_{n_layer // n_group}, ...)
    (Attn_1, Attn_{1 + n_layer // n_group}, ...), (Expert_1, Expert_{1 + n_layer // n_group}, ...)
    ```
    """
    
    @override
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        tp_size = self.model_config.tp_size
        
        assert n_node * n_gpu_per_node % (self.model_config.ep_size + tp_size) == 0
        n_group = n_node * n_gpu_per_node // (self.model_config.ep_size + tp_size)

        node_iter = Counter()
        attn_devs = []
        exp_devs = []
        device_groups = {}
        for i in range(n_group):
            for j in range(tp_size):
                attn_devs.append(next(node_iter))
            devs = attn_devs[-tp_size:]
            for dev in devs:
                device_groups[dev] = devs
            layer_exp_devs = []
            for j in range(self.model_config.ep_size):
                exp_dev = next(node_iter)
                layer_exp_devs.extend([exp_dev] * self.model_config.num_experts_per_rank)
            exp_devs.append(layer_exp_devs)
        
        attn = {attn_dev: [] for attn_dev in attn_devs}
        expert = {}
        for tp in exp_devs:
            for exp_dev in tp:
                expert[exp_dev] = []
        
        pg = ModelPlacement(
            attn, expert, {}, {}, device_groups
        )
        
        for i in range(n_layer):
            # attn driver
            attn_driver = attn_devs[i % n_group * tp_size]
            # all attn workers
            for j in range(tp_size):
                attn[attn_driver + j].append(i)
                
            for j in range(n_expert):
                exp_dev = exp_devs[i % n_group][j]
                expert[exp_dev].append((i, j))
        
        return pg
        

class PipelinePlacement(PlacementBase):
    """
        Parameters: p=step_attn, q=step_exp
    
        First we get virtual mapping:
    
        V_0:        [Attn_0, Attn_p, Attn_{2p}, ...]
        V_1:        [Attn_1, Attn_{p+1}, Attn_{2p+1}, ...]
        ...
        V_{p-1}:    [Attn_{p-1}, Attn_{2p-1}, Attn_{3p-1}, ...]
        
        V_p:        [Expert_0, Expert_q, Expert_{2q}, ...]
        V_{p+1}:    [Expert_1, Expert_{q+1}, Expert_{2q+1}, ...]
        ...
        V_{p+q-1}:  [Expert_{q-1}, Expert_{2q-1}, Expert_{3q-1}, ...]
        
        The index here for Attn/Expert **stands for the layer id**.
        
        We have:
            [V_0, V_{p-1}] -> [G_0, G_{p * TP_SIZE * DP_SIZE - 1}]
            [V_p, V_{p+q-1}] -> [G_{p * TP_SIZE * DP_SIZE}, G_{p * TP_SIZE * DP_SIZE + q * EP_SIZE - 1}]
        
        Ideally, we need a physical mapping to minimize the cross-node communication.
        TODO(hogura|20241212): leave this auto optimization as future work.
    
    """
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig, 
                 step_attn: int, step_expert: int, 
                 zigzag_attn: bool = True, expert_allocation: Optional[List[int]] = None):
        super().__init__(model_config, cluster_config, expert_allocation=expert_allocation)
        self.step_attn = step_attn
        self.step_expert = step_expert
        self.zigzag_attn = zigzag_attn
    
    def _solve_virtual(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        p = self.step_attn
        q = self.step_expert
        
        attns = {
            i: [] for i in range(p)
        }
        experts = {
            i: [] for i in range(q)
        }
        
        for i in range(self.num_layers):
            if self.zigzag_attn:
                attn_dev = i % p
            else:
                attn_dev = i // (self.num_layers // p)
            
            exp_dev = i % q

            attns[attn_dev].append(i)
            experts[exp_dev].append(i)
            
        return ModelPlacement(attns, experts, -1, -1, {}, {})
    
    def _solve_physical(self, mp: ModelPlacement) -> ModelPlacement:
        p = self.step_attn
        q = self.step_expert
        
        num_attn_workers = p * self.tp_size * self.dp_size
        
        attns = {
            i: [] for i in range(num_attn_workers)
        }
        experts = {
            i: [] for i in range(num_attn_workers, 
                                 num_attn_workers + q * self.ep_size)
        }
        device_groups = {
            i: list(range(i * self.tp_size, (i + 1) * self.tp_size)) for i in range(p * self.dp_size)
        }
        
        """
            TODO(hogura|20241222): consider the cross-node communication as follows
            [V_0, V_{p-1}] -> [G_0, G_{p * TP_SIZE * DP_SIZE - 1}]
                1. DP in different nodes
                    2. stride
                        3. TP in the same node
            [V_p, V_{p+q-1}] -> [G_{p * TP_SIZE * DP_SIZE}, G_{p * TP_SIZE * DP_SIZE + q * EP_SIZE - 1}]
                1. ...
        """
        
        for i, layers in mp.attn.items():
            for j in range(self.dp_size):
                attns[(i * self.dp_size + j) * self.tp_size].extend(layers)
        
        for i, layers in mp.expert.items():
            for e in range(self.model_config.num_experts):
                # expert rank: #E -> #E // num_experts_per_rank
                dev_id = num_attn_workers + i * self.ep_size + e // self.model_config.num_experts_per_rank
                for l in layers:
                    experts[dev_id].append((l, e))
        
        return ModelPlacement(
            attns, experts, {}, {},
            device_groups=device_groups
        )
    
    @override
    def _update_expert_rank(self, place: ModelPlacement) -> ModelPlacement:
        if self.expert_allocation is not None:
            expert_ranks = {}
            for dev_id, layers_experts in place.expert.items():
                for layer_id, expert_id in layers_experts:
                    expert_ranks[(layer_id, expert_id)] = dev_id
            place.expert_ranks = expert_ranks
        else:
            expert_ranks = {
                (layer_id, expert_id): expert_id // self.model_config.num_experts_per_rank
                    for layer_id, expert_id in product(range(self.num_layers),
                                                    range(self.model_config.num_experts))
            }
            place.expert_ranks = expert_ranks
        return place
        
    @override
    def _update_attn_dp_rank(self, place: ModelPlacement) -> ModelPlacement:
        attn_dp_ranks = {}
        for dev_id in place.attn:
            attn_dp_ranks[dev_id] = (dev_id // self.tp_size) % self.dp_size
        place.attn_dp_ranks = attn_dp_ranks
        return place
        
    @override
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        n_gpus = n_node * n_gpu_per_node
        p = self.step_attn
        q = self.step_expert
        assert n_gpus >= p * self.tp_size * self.dp_size + q * self.ep_size, \
            f"No enough GPUs for the placement, " \
            f"requiring {p * self.tp_size * self.dp_size + q * self.ep_size}, " \
            f"but only {n_gpus} available."
        mp = self._solve_virtual()
        mp = self._solve_physical(mp)
        return mp

class ColocatePlacement(PlacementBase):
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig, expert_allocation: Optional[List[int]] = None):
        super().__init__(model_config, cluster_config, expert_allocation=expert_allocation)
        
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        num_devices = n_node * n_gpu_per_node
        
        all_layers = list(range(n_layer))
        attns = { i: [] for i in range(num_devices) }
        experts = { i: [] for i in range(num_devices) }
        
        for i in range(num_devices):
            attns[i].extend(all_layers)
        
        if self.expert_allocation is not None:
            assert len(self.expert_allocation) <= num_devices, \
                f"Expert allocation size {len(self.expert_allocation)} exceeds number of devices {num_devices}"
            
            expert_id = 0
            for dev_id, count in enumerate(self.expert_allocation):
                for _ in range(count):
                    for layer_id in all_layers:
                        experts[dev_id].append((layer_id, expert_id))
                    expert_id += 1
            assert expert_id == n_expert
        else:
            for i in range(n_expert):
                for j in all_layers:
                    experts[i // self.model_config.num_experts_per_rank].append((j, i))
        
        device_groups = {
            i: [i] for i in range(num_devices)
        }
        
        local_expert_counts = {}
        for dev_id in range(num_devices):
            local_expert_counts[dev_id] = len(set(eid for _, eid in experts[dev_id]))
        
        return ModelPlacement(
            attns, experts, 
            {}, {}, device_groups=device_groups,
            local_expert_counts=local_expert_counts,
            is_hybrid=True
        )
        
    @override
    def _update_expert_rank(self, place: ModelPlacement) -> ModelPlacement:
        if self.expert_allocation is not None:
            expert_ranks = {}
            for dev_id, layers_experts in place.expert.items():
                for layer_id, expert_id in layers_experts:
                    expert_ranks[(layer_id, expert_id)] = dev_id
            place.expert_ranks = expert_ranks
        else:
            expert_ranks = {
                (layer_id, expert_id): expert_id // self.model_config.num_experts_per_rank
                    for layer_id, expert_id in product(range(self.num_layers),
                                                    range(self.model_config.num_experts))
            }
            place.expert_ranks = expert_ranks
        return place
        
    @override
    def _update_attn_dp_rank(self, place: ModelPlacement) -> ModelPlacement:
        place.attn_dp_ranks = {dev_id: dev_id for dev_id in place.attn}
        return place

_placement_cls: Dict[str, PlacementBase] = {
    "colocate": ColocatePlacement,
    "single": SinglePlacement,
    "interleave": InterleavePlacement,
    "pipeline": PipelinePlacement,
}

def get_model_placement(
    model_config: ModelConfig,
    cluster_config: ClusterConfig,
    strategy: str = "single",
    *args, **kwargs
) -> ModelPlacement:
    if strategy in _placement_cls:
        cls = _placement_cls[strategy]
    else:
        raise NotImplementedError()
    
    placement_kwargs = dict(kwargs)
    placement_kwargs.pop("expert_allocation", None)

    if strategy == "pipeline":
        solver = cls(model_config, cluster_config, *args, **placement_kwargs)
    elif strategy == "colocate":
        solver = cls(model_config, cluster_config, expert_allocation=kwargs.get("expert_allocation", None))
    else:
        solver = cls(model_config, cluster_config, *args, **placement_kwargs)
        
    place: ModelPlacement = solver.solve()
    
    return place
