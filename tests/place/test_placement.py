from disagmoe.utils.placement import *
from disagmoe.config import duo_expert_mixtral
from disagmoe.utils.constants import *

config = duo_expert_mixtral
config.num_layers = 1
config.ep_size = 2
config.num_experts = 2

mp = get_model_placement(
    config,
    ClusterConfig(1, 3, 40 * GiB, TOKENIZER_DEV_ID, SAMPLER_DEV_ID),
    strategy="interleave"
)

print(mp)