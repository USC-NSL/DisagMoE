from disagmoe.ops.memory import permute_tokens_triton, permute_tokens_cuda, get_mappings_from_exp_ids

import torch

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

num_tokens = 64
hidden_size = 4096
num_experts = 8

t = torch.randn((num_tokens, hidden_size))

exp_ids = torch.randint(2, num_experts, (num_tokens, ), device="cpu")
print(f"exp_ids {exp_ids}")

std = torch.empty_like(t)

mappings, cnt = get_mappings_from_exp_ids(exp_ids, num_experts)

print(f"expert cnt {cnt}")
print(f"mappings {mappings}")

pt_triton = permute_tokens_triton(t, mappings)
pt_cuda = permute_tokens_cuda(t, mappings)

for i in range(num_tokens):
    std[mappings[i], :] = t[i, :]

assert(torch.allclose(pt_triton, std))
assert(torch.allclose(pt_cuda, std))