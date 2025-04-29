from benchmark.plotter.namer import add_args, get_plot_dir
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

parser = ArgumentParser()
parser = add_args(parser)
parser.add_argument('-b', '--bin-size', type=int, default=10)
parser.add_argument('-a', '--attn-workers', type=int, default=4)
args = parser.parse_args()

data_path = f"{args.path}/queue_length.pkl"

plot_dir = get_plot_dir(args.path)

bins = {}
    
def put_to_bin(size):
    size = size // args.bin_size
    if size not in bins:
        bins[size] = 0
    bins[size] += 1

def calculate_bin(data):
    # enumerate queue_length as dict
    queue_length, step_executed_layer, step_start_time_ms = data
    
    step_start_time_ms_sampled = np.array(step_start_time_ms)
    step_start_time_ms = step_start_time_ms_sampled - step_start_time_ms_sampled[0]
    for i, l in enumerate(step_executed_layer):
        if l >= 0 and step_start_time_ms[i] > 60 * 1000:
            put_to_bin(queue_length[l][i])
    
def draw_histogram(bin, worker_type):
    plt.figure(figsize=(10, 8))
    plt.bar(bin.keys(), bin.values(), width=1, align='center')
    plt.xlabel(f'Batch Size (by {args.bin_size})')
    plt.ylabel('Frequency')
    plt.title('Batch Size Distribution')
    
    max_ticks = max(bin.keys())
    xticks = range(0, max_ticks + 1, 5)
    xtick_labels = [x * args.bin_size for x in xticks]
    plt.xticks(xticks, xtick_labels, rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{plot_dir}/{worker_type}_batch_size_distribution.png")    

with open(data_path, "rb") as f:
    data = pickle.load(f)

for i in range(args.attn_workers):
    calculate_bin(data[i])
draw_histogram(bins, "attn")

bins = {}
for i in range(args.attn_workers, len(data)):
    calculate_bin(data[i])
draw_histogram(bins, "expert")
