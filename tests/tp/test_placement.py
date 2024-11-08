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
model_config.num_layers = 1

cluster_config = ClusterConfig(
    n_node=1,
    n_gpu=3,
    id_sampler=SAMPLER_DEV_ID,
    id_tokenizer=TOKENIZER_DEV_ID,
)

mp = get_model_placement(model_config, cluster_config, strategy="interleave")

print(mp)