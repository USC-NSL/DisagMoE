from disagmoe.frontend.engine import Engine, Metadata, EngineType
from disagmoe.utils.constants import *
from disagmoe.config import *

from disagmoe_c import Metadata as Metadata_C

import torch

torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.bfloat16)

engine = Engine()

model_config = duo_expert_mixtral
model_config.ep_size = 2
model_config.num_layers = 2
model_config.num_experts = 2

engine.set_device_id(0)
engine.setup_engine(
    EngineType.EXPERT,
    model_config
)

shape = [7, model_config.hidden_size]

meta: Metadata = Metadata_C(shape)
meta.layer_id = 0
meta.req_ids = [0, 1, 2, 3, 4, 5, 6]
meta.exp_ids = [0, 0, 0, 0, 0, 0, 0]
meta.init_prefill_lens = [0, -1, 0, -1, -1, 0, 0]

tensor = torch.zeros(shape)

output, new_meta = engine.process_batch_expert(meta, tensor)

print(output, new_meta)
print(new_meta.req_ids, new_meta.exp_ids, new_meta.init_prefill_lens)

assert meta.init_prefill_lens == [0, 0, 0, 0, -1, -1, -1]