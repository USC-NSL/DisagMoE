from disagmoe.frontend.controller import init_controller
from disagmoe.utils.placement import ModelPlacement
from disagmoe.utils.constants import *
from disagmoe.config import ModelConfig, CacheConfig, duo_expert_mixtral

import time
import torch

master = init_controller(1, 3)

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

mp = ModelPlacement(
    attn = {
        0: [0, 1, 2],
    },
    expert = {
        1: [(0, 0), (1, 0), (2, 0)],
        2: [(0, 1), (1, 1), (2, 1)],
    },
    tokenizer = tokenizer,
    sampler = sampler,
    in_device_ids = {},
    out_device_ids = {},
)

edges = [
    (tokenizer, 0),
    (0, 1),
    (0, 2),
    (1, sampler),
    (2, sampler),
    
    (sampler, 0),
    (1, 0),
    (2, 0),
]

for edge in edges:
    mp.add_edge(edge[0], edge[1])

model_config = duo_expert_mixtral
cache_config = CacheConfig(BLOCK_SIZE, 0.8, 2, "auto", 
                            num_gpu_blocks=NUM_BLOCKS, 
                            num_reserved_blocks=RESERVED_BLOCKS)

master.init_engine(mp, model_config, cache_config)

print("engine inited")

master.start_engine()

master.start_profile()

print("engine started")

n = 1

master.put_multi_request(n)

stats = master.wait_for_requests(n)

master.stop_workers()
print(">>> Slo Stats:", stats)
