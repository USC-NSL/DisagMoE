from disagmoe.frontend.engine import Engine
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from disagmoe.config import ModelConfig, mixtral_config, CacheConfig

from disagmoe_c import BlockManager as BlockManager_C, AttentionScheduler as AttentionScheduler_C

import torch
import logging

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

torch.set_default_tensor_type(torch.BFloat16Tensor)
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
meta = AttentionBatchMetadata(
    0, 
    shape,
    "fp16",
    bs,
    bs,
    0,
    range(bs),
    [1] * bs,
    [1] * bs,
    []
)

engine.start_profile()

attn_meta = engine._pack_flash_attn_metadata(meta.to_c())

engine.stop_profile()