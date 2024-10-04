from disagmoe.utils.constants import *
from disagmoe.utils.utils import Counter

from typing import Dict, Tuple, Optional, Union, List, override
from dataclasses import dataclass


@dataclass
class ModelPlacement:
    # device_id -> layer_id
    attn: Dict[int, List[int]]
    
    # device_id -> (layer_id, expert_id)
    expert: Dict[int, List[Tuple[int, int]]]
    tokenizer: int
    sampler: int
    
    in_device_ids: Dict[int, List[int]]
    out_device_ids: Dict[int, List[int]]
    
    def attn_ids_at(self, device_id: int) -> List[int]:
        return self.attn.get(device_id, [])
    
    def expert_ids_at(self, device_id: int):
        if device_id in self.expert:
            return [e[1] for e in self.expert[device_id]]
        else:
            return []
        
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
    
        
@dataclass
class ModelConfig:
    n_layer: int = 16
    n_local_expert: int = 8
    n_expert_per_token: int = 1
    
    mem_attn = 1 * GiB
    mem_expert = 100 * MiB


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
        n_layer, n_expert = self.model_config.n_layer, self.model_config.n_local_expert
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
    
    @override
    def solve(self) -> ModelPlacement:
        n_layer, n_expert = self.model_config.n_layer, self.model_config.n_local_expert
        n_node, n_gpu = self.cluster_config.n_node, self.cluster_config.n_gpu
        # use one node for tokenizer & sampler

        n_group = n_node * n_gpu // (1 + n_expert)

        node_iter = Counter()
        attn_devs = tuple(next(node_iter) for _ in range(n_group))
        exp_devs = tuple(tuple(next(node_iter) for _ in range(n_expert)) for i in range(n_group))
        tokenizer = self.cluster_config.id_tokenizer
        sampler = self.cluster_config.id_sampler
        
        attn = {attn_dev: [] for attn_dev in attn_devs}
        expert = {}
        for tp in exp_devs:
            for exp_dev in tp:
                expert[exp_dev] = []
        
        pg = ModelPlacement(
            attn, expert, tokenizer, sampler, {}, {}
        )
        
        last_experts = []
        for i in range(n_layer):
            attn_dev = attn_devs[i % n_group]
            attn[attn_dev].append(i)
            if i == 0:
                pg.add_edge(tokenizer, attn_dev)
            for e in last_experts:
                pg.add_edge(e, attn_dev)
            last_experts = []
            # NOTE(hogura|20240904): here assume ep == n_local_expert
            for j in range(n_expert):
                exp_dev = exp_devs[i % n_group][j]
                expert[exp_dev].append((i, j))
                pg.add_edge(attn_dev, exp_dev)
                last_experts.append(exp_dev)
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
    for dev, layer_ids in place.attn.items():
        if 0 in layer_ids:
            place.add_edge(
                place.sampler,
                dev
            )
    
    return place
