from disagmoe.ops.memory import permute_tokens, get_mappings_from_exp_ids

import torch

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

num_tokens = 16
hidden_size = 2048
num_experts = 8

t = torch.randn((num_tokens, hidden_size))

exp_ids = torch.randint(2, num_experts, (num_tokens, ), device="cpu")
print(f"exp_ids {exp_ids}")

std = torch.empty_like(t)

mappings, cnt = get_mappings_from_exp_ids(exp_ids, num_experts)

pt, _ = permute_tokens(t, exp_ids, num_experts)

print(f"expert cnt {cnt}")

for i in range(num_tokens):
    std[mappings[i], :] = t[i, :]

print(f"original tokens {t}")
print(f"permuted tokens {pt}")
assert(torch.allclose(pt, std))