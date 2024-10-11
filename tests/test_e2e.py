from disagmoe.frontend.controller import init_controller
from disagmoe.utils.placement import ModelPlacement
from disagmoe.utils.constants import *

import time

master = init_controller(1, 2)

# 2(tokenizer) -> 0(attn) -> 1(expert) -> 3(sampler)

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

mp = ModelPlacement(
    attn = {
        0: [0],
    },
    expert = {
        1: [(0, 0)],
    },
    tokenizer = tokenizer,
    sampler = sampler,
    in_device_ids = {
        0: [sampler, tokenizer],
        1: [0],
        sampler: [1],
        tokenizer: [],
    },
    out_device_ids = {
        0: [1],
        1: [sampler],
        sampler: [0],
        tokenizer: [0],
    }
)

master.init_engine(mp)

print("engine inited")

master.start_engine()

print("engine started")

input()

master.put_request([1])

while True:
    pass