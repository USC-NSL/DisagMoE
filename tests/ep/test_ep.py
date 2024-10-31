from disagmoe.frontend.controller import init_controller
from disagmoe.utils.placement import ModelPlacement, ClusterConfig, get_model_placement
from disagmoe.utils.constants import *
from disagmoe.config import ModelConfig, CacheConfig, duo_expert_mixtral

import time
import torch

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

cluster_config = ClusterConfig(n_node=1, n_gpu=6, 
                               id_tokenizer=tokenizer, 
                               id_sampler=sampler)

master = init_controller(cluster_config.n_node, cluster_config.n_gpu)

model_config = duo_expert_mixtral
model_config.num_layer = 16
model_config.ep_size = 2
model_config.num_experts = 8

mp = get_model_placement(model_config, cluster_config, "interleave")

print(mp)

cache_config = CacheConfig(BLOCK_SIZE, 0.8, 2, "auto", 
                            num_gpu_blocks=NUM_BLOCKS, 
                            num_reserved_blocks=RESERVED_BLOCKS)

master.init_engine(mp, model_config, cache_config)

print("engine inited")

master.start_engine()

master.start_profile()

print("engine started")

n = 32

master.put_multi_request(n)

stats = master.wait_for_requests(n)

master.stop_workers()
print(">>> Slo Stats:", stats)
