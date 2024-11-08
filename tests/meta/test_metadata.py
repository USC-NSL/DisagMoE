from disagmoe.frontend.engine import Engine
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from disagmoe.config import ModelConfig, mixtral_config, CacheConfig
from disagmoe.utils.utils import Counter

from disagmoe_c import BlockManager as BlockManager_C, AttentionScheduler as AttentionScheduler_C

import torch
import logging

import cProfile

model_config = mixtral_config
cache_config = CacheConfig(
    block_size=32,
    gpu_memory_utilization=0.9,
    swap_space=0,
    cache_dtype="auto",
    num_gpu_blocks=4096,
    num_reserved_blocks=1024,
)
bs = 256

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda:0")

engine = Engine()
engine.device_id = 0
engine._logger = logging.getLogger("engine")

engine.block_mgr = BlockManager_C(
    cache_config.block_size, 
    cache_config.num_gpu_blocks, 
    cache_config.num_reserved_blocks)

shape = (bs, model_config.hidden_size)
tensor = torch.zeros(shape, dtype=torch.bfloat16).cuda()
counter = Counter()

for i in range(3):
    meta = AttentionBatchMetadata(
        0, 
        shape,
        "fp16",
        bs,
        bs,
        0,
        [next(counter) for i in range(bs)],
        [1] * bs,
        [1] * bs,
        []
    )
    engine._pack_flash_attn_metadata(meta.to_c())

def main():
    meta = AttentionBatchMetadata(
        0, 
        shape,
        "fp16",
        bs,
        bs,
        0,
        [next(counter) for i in range(bs)],
        [1] * bs,
        [1] * bs,
        []
    )
    engine._pack_flash_attn_metadata(meta.to_c())

# engine.start_profile()

# cProfile.run("main()", sort="cumulative")
main()

# engine.stop_profile()