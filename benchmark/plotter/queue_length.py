from benchmark.plotter.namer import get_queue_length_name, get_plot_dir
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import os

parser = ArgumentParser()
parser.add_argument('--rate', type=float, required=True)
parser.add_argument('--num-nodes', type=int, default=1)
parser.add_argument('--dp-size', type=int, default=1)
parser.add_argument('--ep-size', type=int, default=1)
parser.add_argument('--steps', type=int, default=100)

args = parser.parse_args()

plot_dir = f"{get_plot_dir(args)}/queue_length_over_time/"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

with open(get_queue_length_name(args), "rb") as f:
    data = pickle.load(f)
    
def sample_from_mid(ls, steps):
    if len(ls) <= steps:
        return ls
    start = (len(ls) - steps) // 2
    return ls[start:start+steps]

def draw_plot(worker_id, queue_length):
    # enumerate queue_length as dict
    queue_length = dict(queue_length)
    plt.figure(figsize=(10, 5))
    
    for layer_id, length in queue_length.items():
        nsteps = len(length)
        plt.plot(sample_from_mid(list(range(nsteps)), args.steps), sample_from_mid(length, args.steps), '-', label=f'layer={layer_id}')
        
    plt.xlabel('step')
    plt.ylabel('queue length')
    plt.legend()
    plt.title(f'queue length per layer (Rate={args.rate}, Nodes={args.num_nodes})')
    plt.savefig(f'{plot_dir}/{worker_id}.png')
    plt.close()

# for each row of df, do draw_plot
for i in range(len(data)):
    draw_plot(i, data[i])

