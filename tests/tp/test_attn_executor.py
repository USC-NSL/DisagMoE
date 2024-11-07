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

master = init_controller(cluster_config.n_node, cluster_config.n_gpu)

master.init_engine(mp, model_config)

attn_workers = [w for w in master.workers if ray.get(w._is_attn.remote())]

assert len(attn_workers) == 2

w0 = attn_workers[0]
w1 = attn_workers[1]

shape = (1, model_config.hidden_size)
tensor = torch.Tensor(shape, device="cuda:0", dtype=torch.bfloat16)
meta = AttentionBatchMetadata(
    [0], 
    shape,
    "fp16",
    1,
    1,
    0,
    [0],
    [1],
    [1],
    []
)

print("Processing batch")

results = ray.get([
    w0.process_batch_attn.remote(tensor, meta),
    w1.process_batch_attn.remote(tensor, meta)
])

print(results)