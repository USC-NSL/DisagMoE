from disagmoe.frontend.controller import init_controller
from disagmoe.utils.placement import ModelPlacement
from disagmoe.utils.constants import *

import time

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

master.init_engine(mp)

print("engine inited")

master.start_engine()

print("engine started")

n = 1

master.put_multi_request(n)

stats = master.wait_for_requests(n)

print(">>> Slo Stats:", stats)
