from disagmoe.utils.placement import get_model_placement, ClusterConfig
from disagmoe.config import ModelConfig, CacheConfig, mixtral_config

model_config = mixtral_config
model_config.dp_size = 4
model_config.tp_size = 1
model_config.ep_size = 4
model_config.num_layers = 4
model_config.num_experts = 8

cluster_config = ClusterConfig(1, 8, id_tokenizer="T", id_sampler="S")

mp = get_model_placement(model_config, cluster_config, "pipeline", step_attn=1, step_expert=1)
