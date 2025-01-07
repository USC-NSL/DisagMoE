from disagmoe.ops.memory import permute_tokens_cuda as permute_tokens, get_mappings_from_exp_ids, get_mappings_from_exp_ids_cuda

import torch

bs = 256
hs = 4096
topk = 1
n_experts = 8
T = 10

torch.manual_seed(1)

exp_ids = torch.randint(0, n_experts, (bs, topk)).to("cuda")
tokens = torch.randn((bs, hs)).to("cuda").to(torch.bfloat16)

print(exp_ids)

# warmup
for _ in range(2):
    _ = permute_tokens(tokens, exp_ids.view((bs, )).tolist())

torch.cuda.synchronize()

### CPU test

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
exp_ids_list = exp_ids.view((bs, )).tolist()

for _ in range(T):
    mappings_cpu, _ = get_mappings_from_exp_ids(exp_ids_list, n_experts)
    _ = permute_tokens(tokens, mappings_cpu)
    torch.cuda.synchronize()
end.record()
torch.cuda.synchronize()

print(f"CPU get_mappings time: {start.elapsed_time(end) / T} ms")

### GPU test

# warmup
for _ in range(5):
    get_mappings_from_exp_ids_cuda(exp_ids, n_experts)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for _ in range(T):
    mappings_gpu, _ = get_mappings_from_exp_ids_cuda(exp_ids, n_experts)
    _ = permute_tokens(tokens, mappings_gpu)
end.record()

torch.cuda.synchronize()

print(f"GPU get_mappings time: {start.elapsed_time(end) / T} ms")

### Compare results
print(mappings_cpu)
print(mappings_gpu.view(-1).tolist())
