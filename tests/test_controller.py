from disagmoe.frontend.controller import init_controller
from disagmoe.utils.placement import ModelPlacement

master = init_controller(1, 2)

# 2(tokenizer) -> 0(attn) -> 1(expert) -> 3(sampler)
mp = ModelPlacement(
    attn = {
        0: [0],
    },
    expert = {
        1: [(0, 0)],
    },
    tokenizer = 2,
    sampler = 3,
    in_device_ids = {
        0: [],
        1: [0],
    },
    out_device_ids = {
        0: [1],
        1: [],
    }
)

master.init_engine(mp)

print("passed")

while True:
    pass