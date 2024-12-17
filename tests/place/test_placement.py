from disagmoe.utils.placement import *
from disagmoe.config import mixtral_config
from disagmoe.utils.constants import *

config = mixtral_config
config.num_layers = 4
config.ep_size = 2
config.tp_size = 1
config.num_experts = 4

mp = get_model_placement(
    config, 
    ClusterConfig(1, 8, 40 * GiB, TOKENIZER_DEV_ID, SAMPLER_DEV_ID),
    strategy="pipeline",
    step_attn=2,
    step_expert=1,
    zigzag_attn=True,
)

print(mp)

mp = get_model_placement(
    config, 
    ClusterConfig(1, 8, 40 * GiB, TOKENIZER_DEV_ID, SAMPLER_DEV_ID),
    strategy="pipeline",
    step_attn=2,
    step_expert=1,
    zigzag_attn=False,
)

print(mp)

mp = get_model_placement(
    config, 
    ClusterConfig(1, 8, 40 * GiB, TOKENIZER_DEV_ID, SAMPLER_DEV_ID),
    strategy="pipeline",
    step_attn=1,
    step_expert=1,
    zigzag_attn=True,
)

print(mp)

