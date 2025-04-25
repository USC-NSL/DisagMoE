from disagmoe.utils.constants import *
from disagmoe.utils.utils import Counter
from disagmoe.config import ModelConfig

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
    def from_c(tp: int, ep: int, dp: int, n_exp_per_rank: int, expert_ranks: List) -> "ParallelConfig_C":
        from disagmoe_c import ParallelConfig as ParallelConfig_C
        cfg = ParallelConfig_C()
        cfg.tp = tp
        cfg.ep = ep
        cfg.dp = dp
        cfg.n_exp_per_rank = n_exp_per_rank
        cfg.expert_ranks = expert_ranks
        return cfg

@dataclass
class ModelPlacement:
    # device_id -> layer_id
    attn: Dict[int, List[int]]
    
    # device_id -> (layer_id, expert_id)
    expert: Dict[int, List[Tuple[int, int]]]
    tokenizer: int
    sampler: int
    
    # for the devices in a TP group, only the driver's device_id is stored in the edges
    in_device_ids: Dict[int, List[int]]
    out_device_ids: Dict[int, List[int]]
    
    device_groups: Dict[int, List[int]] = None
    
    # (layer_id, expert_id) -> expert_rank_id
    expert_ranks: Dict[Tuple[int, int], int] = None
    
    # device_id -> attn_rank_id
    attn_dp_ranks: Dict[int, int] = None
    
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
    
    def attn_dp_rank_at(self, device_id: int) -> int:
        return self.attn_dp_ranks.get(device_id, 0)
    
    def layer_ids_at(self, device_id: int) -> List[int]:
        return sorted(list(set(
            self.attn.get(device_id, []) + [e[0] for e in self.expert.get(device_id, [])]
        )))
        
    def add_edge(self, start, end):
        # assert start != end
        if end not in self.in_device_ids:
            self.in_device_ids[end] = []
        if start not in self.out_device_ids:
            self.out_device_ids[start] = []
        if start not in self.in_device_ids[end]:
            self.in_device_ids[end].append(start)
            self.out_device_ids[start].append(end)
            
    def is_worker_device(self, device_id: int) -> bool:
        return device_id in self.device_groups and self.device_groups[device_id][0] != device_id
    
    def has_attn(self, device_id: int) -> bool:
        return device_id in self.attn
    
    def in_device_ids_at(self, device_id: int, tp_enable_inter_group: bool) -> List[int]:
        if self.is_worker_device(device_id) and tp_enable_inter_group:
            # return the driver's in_device_ids
            return [w for w in self.in_device_ids.get(self.device_groups[device_id][0], []) if w not in [self.tokenizer, self.sampler]]
        else:
            return self.in_device_ids.get(device_id, [])

@dataclass
class ClusterConfig:
    n_node: int
    n_gpu: int
    gpu_cap: float = 40 * GiB
    id_tokenizer: int = -1
    id_sampler: int = -1

class PlacementBase:
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig, 
                 step_attn: int = 0, step_expert: int = 0, 
                 zigzag_attn: bool = True):
        self.model_config = model_config
        self.cluster_config = cluster_config
        
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
            if layer_id == 0:
                # tokenizer to the first layer
                for dev in attn_devs[layer_id]:
                    place.add_edge(place.tokenizer, dev)
                    
                # add edges from sampler back to the first attn
                for dev in attn_devs[layer_id]:
                    if self.model_config.tp_size > 1 and place.is_worker_device(dev):
                        # if TP is enabled, the sampler should only connect to driver attn
                        continue
                    place.add_edge(place.sampler, dev)
            else:
                # last exp to current attn
                for dev in attn_devs[layer_id]:
                    for prev_dev in exp_devs[layer_id - 1]:
                        place.add_edge(prev_dev, dev)
            # the last layer to sampler
            if layer_id == self.model_config.num_layers - 1:
                for dev in exp_devs[layer_id]:
                    place.add_edge(dev, place.sampler)
            # current attn to current exp
            for dev in attn_devs[layer_id]:
                for exp_dev in exp_devs[layer_id]:
                    place.add_edge(dev, exp_dev)
        return place
        
    def _update_expert_rank(self, place: ModelPlacement) -> ModelPlacement:
        """
            default EP worker rank is `expert_id // num_experts_per_rank` for each expert
        """
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
        print(place)
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
        # tokenizer and sampler do not use gpu.
        assert n_layer * (self.rep_attn + n_expert) <= n_node * n_gpu_per_node
        # not considering gpu_cap yet
        attn = {}
        expert = {}
        node_id = Counter()
        i_tokenizer = self.cluster_config.id_tokenizer
        i_sampler = self.cluster_config.id_sampler
        pg = ModelPlacement(
            attn, expert, i_tokenizer, i_sampler, {}, {}
        )
        i_last_experts = [i_tokenizer]
        for i in range(n_layer):
            attns = []
            for j in range(self.rep_attn):
                i_attn = next(node_id)
                attn[i_attn] = [i]
                # for i_expert in i_last_experts:
                #     pg.add_edge(i_expert, i_attn)
                attns.append(i_attn)
            
            i_last_experts = []
            for j in range(n_expert):
                i_expert = next(node_id)
                i_last_experts.append(i_expert)
                expert[i_expert] = [(i, j)]
                # for i_attn in attns:
                #     pg.add_edge(i_attn, i_expert)
                # if i == n_layer - 1:
                #     pg.add_edge(i_expert, i_sampler)
        assert len(attn) == n_layer * self.rep_attn
        assert len(pg.in_device_ids) == n_layer * (self.rep_attn + n_expert) + 1
        assert len(pg.out_device_ids) == n_layer * (self.rep_attn + n_expert) + 1
        return pg

class InterleavePlacement(PlacementBase):
    
    """
    The structure of the placement is as follows:
    ```
    Tokenizer
    (Attn_0, Attn_{n_layer // n_group}, ...), (Expert_0, Expert_{n_layer // n_group}, ...)
    (Attn_1, Attn_{1 + n_layer // n_group}, ...), (Expert_1, Expert_{1 + n_layer // n_group}, ...)
    ...
    Embedding
    ```
    """
    
    @override
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        tp_size = self.model_config.tp_size
        
        # tokenizer & sampler do not use GPU.
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
        tokenizer = self.cluster_config.id_tokenizer
        sampler = self.cluster_config.id_sampler
        
        attn = {attn_dev: [] for attn_dev in attn_devs}
        expert = {}
        for tp in exp_devs:
            for exp_dev in tp:
                expert[exp_dev] = []
        
        pg = ModelPlacement(
            attn, expert, tokenizer, sampler, {}, {}, device_groups
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
                 zigzag_attn: bool = True):
        super().__init__(model_config, cluster_config)
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
            attns, experts, self.cluster_config.id_tokenizer, self.cluster_config.id_sampler, {}, {},
            device_groups=device_groups
        )
    
    @override
    def _update_expert_rank(self, place: ModelPlacement) -> ModelPlacement:
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
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig):
        super().__init__(model_config, cluster_config)
        
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        num_devices = n_node * n_gpu_per_node
        
        all_layers = list(range(n_layer))
        attns = { i: [] for i in range(num_devices) }
        experts = { i: [] for i in range(num_devices) }
        
        for i in range(num_devices):
            attns[i].extend(all_layers)
        
        for i in range(n_expert):
            for j in all_layers:
                experts[i // self.model_config.num_experts_per_rank].append((j, i))
        
        device_groups = {
            i: [i] for i in range(num_devices)
        }
        
        return ModelPlacement(
            attns, experts, self.cluster_config.id_tokenizer, self.cluster_config.id_sampler, 
            {}, {}, device_groups=device_groups, is_hybrid=True
        )
        
    @override
    def _update_expert_rank(self, place: ModelPlacement) -> ModelPlacement:
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

class CoordinatePlacement(ColocatePlacement):
    """
        Each node is seperated into two groups, [0, gpu_attn) for attn DP, and [gpu_attn, n_gpu) for experts.
    """
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig, gpu_attn: int=-1):
        super().__init__(model_config, cluster_config)
        # self.gpu_attn = gpu_attn if gpu_attn != -1 else cluster_config.n_gpu // 2
        self.gpu_attn = cluster_config.n_gpu - model_config.ep_size // cluster_config.n_node
        
    @override
    def _solve(self, n_layer: int, n_expert: int, n_node: int, n_gpu_per_node: int) -> ModelPlacement:
        num_devices = n_node * n_gpu_per_node
        
        all_layers = list(range(n_layer))
        attns = {}
        experts = {}
        
        cnt_expert = 0
        attn_dp_cnt = 0
        attn_dp_ranks = {}
        for i in range(n_node):
            for j in range(n_gpu_per_node):
                dev_id = i * n_gpu_per_node + j
                if j < self.gpu_attn:
                    attns[dev_id] = all_layers
                    attn_dp_ranks[dev_id] = attn_dp_cnt
                    attn_dp_cnt += 1
                else:
                    experts[dev_id] = []
                    for _ in range(self.model_config.num_experts_per_rank):
                        assert cnt_expert < n_expert
                        for layer_id in all_layers:
                            experts[dev_id].append((layer_id, cnt_expert))
                        cnt_expert += 1
        
        device_groups = {
            i: [i] for i in range(num_devices)
        }
        
        return ModelPlacement(
            attns, experts, self.cluster_config.id_tokenizer, self.cluster_config.id_sampler, 
            {}, {},
            attn_dp_ranks=attn_dp_ranks, 
            device_groups=device_groups
        )

    @override
    def _update_attn_dp_rank(self, place: ModelPlacement) -> ModelPlacement:
        return place

_placement_cls: Dict[str, PlacementBase] = {
    "colocate": ColocatePlacement,
    "single": SinglePlacement,
    "interleave": InterleavePlacement,
    "pipeline": PipelinePlacement,
    "coordinate": CoordinatePlacement,
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
    
    if strategy == "pipeline":
        solver = cls(model_config, cluster_config, *args, **kwargs)
    elif strategy in ["colocate", "coordinate"]:
        solver = cls(model_config, cluster_config)
    else:
        raise NotImplementedError()
        
    place: ModelPlacement = solver.solve()
    
    print(f"Model Placement: {place}")
    
    return place
