from grouped_gemm.backend import gmm, gmm_with_arguments, get_arguments
import torch

n_experts = 8
bs = n_experts * 2
hs = 1024

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(torch.device("cuda:0"))

w1 = torch.randn((n_experts, hs, hs))
x = torch.randn((bs, hs))
batch_sizes = torch.tensor([bs // n_experts] * n_experts, dtype=torch.int64)

y = gmm(x, w1, batch_sizes)
# print(y)

print(gmm)

print(gmm_with_arguments)

print(get_arguments)

workspace_size, ptr = get_arguments(n_experts, torch.device("cuda:0"))

print(workspace_size, ptr)

workspace = torch.empty([workspace_size], dtype=torch.uint8, device=torch.device("cuda:0"))

y2 = gmm_with_arguments(x, w1, batch_sizes, workspace, ptr)
y3 = gmm_with_arguments(x, w1, batch_sizes, workspace, ptr)

# print(y2)

assert torch.allclose(y, y2)
assert torch.allclose(y, y3)