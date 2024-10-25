import torch
from disagmoe.models.experts import MoEExperts

hidden_size = 1024
num_experts = 8
intermediate_size = 4 * hidden_size

batch_size = 256

torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.bfloat16)
inputs = torch.rand((batch_size, hidden_size))
dist = torch.rand(num_experts, device="cpu")
total = torch.sum(dist).item()
dist = dist / total * batch_size
dist = dist.to(dtype=torch.int64)
sum_tokens = torch.sum(dist).item()
if sum_tokens < batch_size:
    dist[-1] += batch_size - sum_tokens
    
experts = MoEExperts(hidden_size, intermediate_size, num_experts)
outputs = experts(inputs, dist)
print(outputs)

