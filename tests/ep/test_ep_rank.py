import torch

from disagmoe.frontend.engine import Engine, EngineType
from disagmoe.frontend.datatypes import Metadata
from disagmoe.config import duo_expert_mixtral

from disagmoe_c import Metadata as Metadata_C

cfg = duo_expert_mixtral
cfg.ep_size = 2
cfg.rank = 1

engine = Engine()
engine.set_device_id(0)
engine.setup_engine(
    EngineType.EXPERT,
    cfg,
)
engine.init_core([0], [], [], [], {})

meta: Metadata = Metadata_C([4, cfg.hidden_size])
meta.layer_id = 0
meta.req_ids = [0, 1, 2, 3]
meta.exp_ids = [1, 1, 1, 1]
meta.init_prefill_lens = [-1, -1, -1, -1]
meta.dtype = "bf16"

tensor = torch.randn(meta.shape)

output, new_meta = engine.process_batch_expert(
    meta,
    tensor
)

print(output)
print(new_meta)