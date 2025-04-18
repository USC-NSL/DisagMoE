from disagmoe.utils.placement import *
from disagmoe.config import mixtral_config
from disagmoe.utils.constants import *

config = mixtral_config
config.num_layers = 32
config.ep_size = 2
config.tp_size = 1
config.dp_size = 2
config.num_experts = 4

mp = get_model_placement(
    config, 
    ClusterConfig(1, 4, 40 * GiB, TOKENIZER_DEV_ID, SAMPLER_DEV_ID),
    strategy="coordinate",
)

print(mp)

for i in range(0, 4):
    print(mp.rank_at(i, config.num_experts_per_rank))