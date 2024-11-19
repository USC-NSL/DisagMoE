import ray
import torch

from disagmoe.utils.placement import get_model_placement, ClusterConfig
from disagmoe.config import ModelConfig, CacheConfig, mixtral_config
from disagmoe.utils.constants import *
from disagmoe.frontend.controller import Controller, init_controller
from disagmoe.frontend.datatypes import AttentionBatchMetadata

model_config = mixtral_config
model_config.tp_size = 2
model_config.num_experts = 1
model_config.ep_size = 1
model_config.num_layers = 2

cluster_config = ClusterConfig(
    n_node=1,
    n_gpu=model_config.tp_size + model_config.ep_size,
    id_sampler=SAMPLER_DEV_ID,
    id_tokenizer=TOKENIZER_DEV_ID,
)

cache_config = CacheConfig(
    block_size=32,
    gpu_memory_utilization=0.9,
    swap_space=0,
    cache_dtype="auto",
    num_gpu_blocks=4096,
    num_reserved_blocks=1024,
)

mp = get_model_placement(model_config, cluster_config, strategy="interleave")
print(mp)

master = init_controller(cluster_config.n_node, cluster_config.n_gpu)
master.init_engine(mp, model_config, cache_config)

print("start profile")
master.start_profile()

print("start engine")
master.start_engine()

n = 1

print("put multi request")
master.put_requests([1] * n)

print("wait for request")
stats = master.wait_for_requests(n)
print(">>> Slo Stats:", stats)
