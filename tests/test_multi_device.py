from disagmoe.frontend.controller import init_controller
from disagmoe.utils.placement import ModelPlacement
from disagmoe.utils.constants import *

import time

master = init_controller(1, 4)

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

mp = ModelPlacement(
    attn = {
        0: [0, 2],
        2: [1, 3],
    },
    expert = {
        1: [(0, 0), (2, 0)],
        3: [(1, 0), (3, 0)],
    },
    tokenizer = tokenizer,
    sampler = sampler,
    in_device_ids = {},
    out_device_ids = {},
)

edges = [
    (tokenizer, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, sampler),
    
    (sampler, 0),
    (3, 0),
]

for edge in edges:
    mp.add(edge[0], edge[1])

master.init_engine(mp)

print("engine inited")

master.start_engine()

print("engine started")

time.sleep(2)

master.put_request([1])

while True:
    pass