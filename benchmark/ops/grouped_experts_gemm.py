import torch
from grouped_gemm.backend import gmm
from argparse import ArgumentParser
import numpy as np

def get_args():
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--num_experts", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=4096)
    
    args = parser.parse_args()

    return args

@torch.inference_mode()
def main():

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda:0')
    torch.manual_seed(0)
    args = get_args()

    batch_sizes = torch.rand(args.num_experts, device="cpu")
    batch_sizes = batch_sizes * args.batch_size / sum(batch_sizes) 
    batch_sizes = batch_sizes.to(torch.int64)
    batch_sizes[-1] += args.batch_size - batch_sizes.sum()
    batch_sizes = batch_sizes.to("cuda")

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

        start.record()
        for _ in range(t):
            op(*args)
        end.record()
        torch.cuda.synchronize()

        print(f"{name} time: {start.elapsed_time(end) / t} ms")

    benchmark("grouped_gemm", gmm, inputs, weights, batch_sizes)


    batch_sizes_list = batch_sizes.tolist()

    def baseline():
        for i, b in enumerate(batch_sizes_list):
            inputs_ = inputs[:b]
            weights_ = weights[i]
            torch.matmul(inputs_, weights_)

    benchmark("serial_gemm", baseline)

    print(batch_sizes)

    # prof.stop()
        

if __name__ == '__main__':
    main()