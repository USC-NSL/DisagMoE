import torch
import argparse

from disagmoe.models.experts import MoEExperts, MoEExpertsSerial

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_experts", type=int, default=8)
parser.add_argument("-b", "--batch_size", type=int, default=512)

args = parser.parse_args()

hs = 4096
isize = hs * 4
n_experts = args.num_experts
bs = args.batch_size

torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.bfloat16)

std_weights = MoEExperts(hidden_size=hs, intermediate_size=isize, num_experts=n_experts).state_dict()

def benchmark(expert_cls: MoEExperts, hiddens: torch.Tensor, batch_sizes: torch.Tensor, enable_cuda_graph: bool):
    expert: MoEExperts = expert_cls(hidden_size=hs, intermediate_size=isize, num_experts=n_experts)
    expert.load_state_dict(std_weights)
    
    output = torch.zeros_like(hiddens)
    # warmup
    if enable_cuda_graph:
        sampler = torch.distributions.Multinomial(probs=torch.ones(n_experts), total_count=bs)
        static_batch_sizes = sampler.sample().to(torch.int64)
        static_batch_sizes[-1] = bs - static_batch_sizes[:-1].sum()
        static_hiddens = torch.randn_like(hiddens)
        static_output = torch.zeros_like(hiddens)
    for _ in range(3):
        output = expert.forward(hiddens, batch_sizes)
        
    if enable_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = expert.forward(static_hiddens, static_batch_sizes)
        torch.cuda.synchronize()
        graph.replay()
        
    torch.cuda.synchronize()
    
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    
    STEP = 10
    start.record()
    for _ in range(STEP):
        if enable_cuda_graph:
            static_batch_sizes.copy_(batch_sizes)
            static_hiddens.copy_(hiddens)
            graph.replay()
            output.copy_(static_output)
        else:
            output = expert.forward(hiddens, batch_sizes)
        torch.cuda.synchronize()
    end.record()
    
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / STEP, output

sampler = torch.distributions.Multinomial(probs=torch.ones(n_experts), total_count=bs)
batch_sizes = sampler.sample().to(torch.int64)
batch_sizes[-1] = bs - batch_sizes[:-1].sum()

hiddens = torch.randn(bs, hs)

results = []

for exp_name, exp_cls in [("MoEExperts", MoEExperts), ("MoEExpertsSerial", MoEExpertsSerial)]:
    for enable_cuda_graph in [False, True]:
        t, out = benchmark(exp_cls, hiddens, batch_sizes, enable_cuda_graph)
        results.append(out)
        print(f"{exp_name, enable_cuda_graph}: {t:.2f} ms")
    
for result in results[1:]:
    print(result - results[0])
# assert torch.allclose(*results, rtol=1e-2)