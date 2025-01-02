from disagmoe.frontend.engine import Engine, ExpertsExecutor, EngineType
from disagmoe.config import ModelConfig, mixtral_config
from disagmoe.frontend.datatypes import Metadata
from disagmoe_c import recorder_create, recorder_output

import logging
import torch
import copy

logging.basicConfig(level=logging.INFO)

bs = 256
hs = 4096
T = 30

engine = Engine()

cfg = mixtral_config
cfg.enable_cuda_graph_expert = False
cfg.num_layers = 1
cfg.layer_ids = [0]
cfg.graph_stride = 256
cfg.ep_size = 2
cfg.max_batch_size_expert = 512

engine.model_config = cfg

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

# init engine
engine.device_id = 0
engine._logger = logging.getLogger("engine")
engine.engine_type = EngineType.EXPERT
engine.executor = ExpertsExecutor(engine.model_config)
engine._process_batch = engine.process_batch_expert_optimized
# prepare inner exp rank, [n_exp_per_rank * rank, (rank + 1) * n_exp_per_rank) -> [0, n_exp_per_rank)
engine.inner_exp_rank = [0 for _ in range(engine.model_config.num_experts_per_rank)]
for i in range(engine.model_config.num_experts_per_rank):
    engine.inner_exp_rank[i] = engine.model_config.num_experts_per_rank * engine.rank_in_group + i
engine.max_batch_size = cfg.max_batch_size_expert

print("init engine")

# warmup
recorder_create()
engine._warmup()
engine._create_cuda_graph_contexts()
if engine.model_config.enable_cuda_graph_expert:
    engine._cuda_graph_capture()

print("warmup")

meta = Metadata(
    [bs, hs],
    "bf16",
    0,
    list(range(bs)),
    [i % engine.model_config.num_experts_per_rank for i in range(bs)],
    [0 for i in range(bs)],
    [0 for i in range(bs)],
)

meta_c = meta.to_c()

tensor = torch.randn(bs, hs, dtype=torch.bfloat16, device="cuda")

# warmup again
for _ in range(2):
    engine._process_batch(meta_c, tensor)
    meta_c = meta.to_c()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# engine.start_profile("torch_profile")
start.record()
for _ in range(T):
    output, new_meta = engine._process_batch(meta_c, tensor)
    meta_c = meta.to_c()
    torch.cuda.synchronize()
end.record()
torch.cuda.synchronize()
# engine.stop_profile()

print(f"Time: {start.elapsed_time(end) / T} ms")

_, stats = engine.fetch_step_stats()

events = []
ms_to_us = lambda ms: ms * 1000
for tid, t_traces in stats.items():
    print("outputing thread", tid)
    tid = tid % (1 << 32)
    for trace in t_traces:
        events.append({
            "name": trace.msg,
            "cat": "trace",
            "ph": "X",
            "ts": ms_to_us(trace.t_start),
            "dur": ms_to_us(trace.t_dur),
            "pid": 0,
            "tid": (tid * 10 + trace.track_id) % (1 << 31),
        })

import gzip
import json
with gzip.open(f"optimizing_expert.json.gz", "w") as f:
    f.write(json.dumps(events).encode("utf-8"))