from disagmoe.utils.constants import *
from disagmoe.utils.utils import Counter
from disagmoe.config import ModelConfig

from typing import Dict, Tuple, Optional, Union, List, override
from dataclasses import dataclass

@dataclass
class ParallelConfig:
    tp: int = 1
    ep: int = 1

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
    
    def expert_rank_at(self, device_id: int, num_expert_per_rank: int) -> int:
        assert device_id in self.expert
        ids = self.expert[device_id]
        ranks = []
        for layer_id, expert_id in ids:
            ranks.append(expert_id // num_expert_per_rank)
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
    
    def attn_ids_at(self, device_id: int) -> List[int]:
        return self.attn.get(device_id, [])
    
    def expert_ids_at(self, device_id: int):
        return self.expert.get(device_id, [])
        
    def layer_ids_at(self, device_id: int) -> List[int]:
        return list(set(
            self.attn.get(device_id, []) + [e[0] for e in self.expert.get(device_id, [])]
        ))
        
    def add_edge(self, start, end):
        assert start != end
        if end not in self.in_device_ids:
            self.in_device_ids[end] = []
        if start not in self.out_device_ids:
            self.out_device_ids[start] = []
        if start not in self.in_device_ids[end]:
            self.in_device_ids[end].append(start)
            self.out_device_ids[start].append(end)
            
    def is_worker_device(self, device_id: int) -> bool:
        return device_id in self.device_groups and self.device_groups[device_id][0] != device_id

    def is_attn(self, device_id: int) -> bool:
        # NOTE(hogura|20241111): since the dict `attn` only includes driver now, we use `expert` to check
        return device_id not in self.expert

@dataclass
class ClusterConfig:
    n_node: int
    n_gpu: int
    gpu_cap: float = 40 * GiB
    id_tokenizer: int = -1
    id_sampler: int = -1

class PlacementBase:
    
    def __init__(self, 
                 model_config: ModelConfig,
                 cluster_config: ClusterConfig):
        self.model_config = model_config
        self.cluster_config = cluster_config
        
    def solve(self) -> ModelPlacement:
        raise NotImplementedError()


class SinglePlacement(PlacementBase):
    
    def __init__(self, model_config: ModelConfig, cluster_config: ClusterConfig, rep_attn: int=1):
        super().__init__(model_config, cluster_config)
        self.rep_attn = rep_attn
    
    @override
    def solve(self) -> ModelPlacement:
        n_layer, n_expert = self.model_config.num_layers, self.model_config.num_experts
        n_node, n_gpu = self.cluster_config.n_node, self.cluster_config.n_gpu
        
        # 1 attn, n_expert experts
        # tokenizer and sampler do not use gpu.
        assert n_layer * (self.rep_attn + n_expert) <= n_node * n_gpu
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
                for i_expert in i_last_experts:
                    pg.add_edge(i_expert, i_attn)
                attns.append(i_attn)
            
            i_last_experts = []
            for j in range(n_expert):
                i_expert = next(node_id)
                i_last_experts.append(i_expert)
                expert[i_expert] = [(i, j)]
                for i_attn in attns:
                    pg.add_edge(i_attn, i_expert)
                if i == n_layer - 1:
                    pg.add_edge(i_expert, i_sampler)
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
    def solve(self) -> ModelPlacement:
        n_layer, n_expert = self.model_config.num_layers, self.model_config.num_experts
        n_node, n_gpu = self.cluster_config.n_node, self.cluster_config.n_gpu
        tp_size = self.model_config.tp_size
        
        # tokenizer & sampler do not use GPU.
        assert n_node * n_gpu % (self.model_config.ep_size + tp_size) == 0
        n_group = n_node * n_gpu // (self.model_config.ep_size + tp_size)

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
        print(attn)
        expert = {}
        for tp in exp_devs:
            for exp_dev in tp:
                expert[exp_dev] = []
        
        pg = ModelPlacement(
            attn, expert, tokenizer, sampler, {}, {}, device_groups
        )
        
        last_experts = []
        for i in range(n_layer):
            # attn driver
            attn_driver = attn_devs[i % n_group * tp_size]
            # all attn workers
            for j in range(tp_size):
                attn[attn_driver + j].append(i)
                
            if i == 0:
                pg.add_edge(tokenizer, attn_driver)
            for e in last_experts:
                pg.add_edge(e, attn_driver)
            last_experts = []
            # NOTE(hogura|20240904): here assume ep == n_local_expert
            for j in range(n_expert):
                exp_dev = exp_devs[i % n_group][j]
                expert[exp_dev].append((i, j))
                pg.add_edge(attn_driver, exp_dev)
                last_experts.append(exp_dev)
            if tp_size > 1:
                pass
                
        for e in last_experts:
            pg.add_edge(e, sampler)
        
        return pg

_placement_cls: Dict[str, PlacementBase] = {
    "single": SinglePlacement,
    "interleave": InterleavePlacement,
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

    solver = cls(model_config, cluster_config, *args, **kwargs)
    place: ModelPlacement = solver.solve()
    
    # add edges from sampler back to the first attn
    # NOTE(hogura|20241107): only connect to driver when tp_size > 1
    for dev, layer_ids in place.attn.items():
        if place.is_worker_device(dev):
            continue
        if 0 in layer_ids:
            place.add_edge(
                place.sampler,
                dev
            )
    
    return place
