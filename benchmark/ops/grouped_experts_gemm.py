import torch
from grouped_gemm.backend import gmm
from argparse import ArgumentParser
import numpy as np

def get_args():
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--num_experts", type=int, default=8)
    parser.add_argument("-i", "--iterations", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--skewed", action="store_true", default=False)
    
    args = parser.parse_args()

    return args

@torch.inference_mode()
def main():

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda:0')
    # torch.manual_seed(0)
    args = get_args()

    # batch_sizes: torch.Tensor = torch.rand(args.num_experts, device="cpu")
    # batch_sizes = batch_sizes * args.batch_size / sum(batch_sizes) 
    # batch_sizes = batch_sizes.to(torch.int64)
    # batch_sizes[-1] += args.batch_size - batch_sizes.sum()
    # batch_sizes = batch_sizes.to("cpu")
    
    if not args.skewed:
        dist = torch.distributions.Multinomial(args.batch_size, torch.ones(args.num_experts))
    else:
        dist = torch.distributions.Multinomial(args.batch_size, torch.Tensor([i + 1 for i in range(args.num_experts)]))
    
    batch_sizes = dist.sample().to(torch.int64).to("cpu")
    if batch_sizes.sum() > args.batch_size:
        batch_sizes[batch_sizes.argmax().item()] -= batch_sizes.sum() - args.batch_size
    elif batch_sizes.sum() < args.batch_size:
        batch_sizes[batch_sizes.argmax().item()] += args.batch_size - batch_sizes.sum()
    
    sigma = batch_sizes.to(torch.float32).std().item()
    print("batch sizes sigma: ", sigma)
    print(batch_sizes)

    # prof = torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('.'),
    # ) 
    # prof.start()

    intermidiate_size = 4 * args.hidden_size

    weights = torch.rand((args.num_experts, args.hidden_size, intermidiate_size), device="cuda:0")

    inputs = torch.rand(args.batch_size, args.hidden_size)

    def benchmark(name, op, *args):

        for _ in range(2):
            op(*args)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        t = 10
        torch.cuda.synchronize()
        start.record()
        for _ in range(t):
            op(*args)
        end.record()
        torch.cuda.synchronize()

        avg = start.elapsed_time(end) / t
        print(f"{name} time: {avg} ms")
        return avg

    t_grouped = benchmark("grouped_gemm", gmm, inputs, weights, batch_sizes)


    batch_sizes_list = batch_sizes.tolist()

    def baseline():
        s = 0
        for i, b in enumerate(batch_sizes_list):
            # inputs_ = inputs[s:s + b]
            inputs_ = inputs[:b]
            weights_ = weights[i]
            torch.matmul(inputs_, weights_)
            s += b

    t_serial = benchmark("serial_gemm", baseline)

    # prof.stop()
    return sigma, t_serial, t_grouped, batch_sizes.min().item()
        

if __name__ == '__main__':
    results = []
    for i in range(get_args().iterations):
        _ = main()
        results.append([*_, 0])
    results = np.array(results)
    results[..., -1] = results[..., 1] / results[..., 2]
    print("sigma, t_serial, t_grouped, min_bs, speedup")
    print(results)