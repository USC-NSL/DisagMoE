from benchmark.plotter.namer import get_queue_length_name, get_plot_dir
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

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

def draw_heatmap(worker_id, data):
    # enumerate queue_length as dict
    queue_length, step_start_time_ms = data
    
    plt.figure(figsize=(10, 2))
    figure, ax = plt.subplots()
    
    layer_ids = []
    nsteps = 0
    
    for layer_id, length in queue_length.items():
        layer_ids.append(layer_id)
        nsteps = len(length)
        
    layer_ids.sort()
    nlayers = len(layer_ids)
    
    ax.set_xlim(0, args.steps)
    ax.set_ylim(0, nlayers)
    
    step_ids = np.array(sample_from_mid(list(range(nsteps)), args.steps))
    step_start_time_ms_sampled = np.array(sample_from_mid(step_start_time_ms, args.steps))
    step_start_time_ms_sampled = step_start_time_ms_sampled - step_start_time_ms_sampled[0]
    step_start_time_ms_sampled = np.round(step_start_time_ms_sampled, 1)
    
    data = np.array([sample_from_mid(queue_length[layer_id], args.steps) for layer_id in layer_ids])
    
    plt.imshow(data, cmap='hot', origin='lower', extent=[0, args.steps, 0, nlayers])
    plt.xticks(np.arange(args.steps)[ : : 10], step_start_time_ms_sampled[ : : 10], rotation=45)
    
    executed_layer_ids = np.argmax(data, axis=0)
    for i, layer_id in enumerate(executed_layer_ids):
        plt.plot([i, i+1], [layer_id, layer_id], color='cyan', linestyle='--', linewidth=1)
    
    plt.xlabel('time (ms)')
    plt.ylabel('layer')
    plt.title(f'queue length per layer (Rate={args.rate}, Nodes={args.num_nodes})')
    plt.colorbar(label='Queue Length', orientation='vertical', shrink=0.4)
    plt.savefig(f'{plot_dir}/{worker_id}.pdf', bbox_inches='tight')
    plt.close()

# for each row of df, do draw_plot
for i in range(len(data)):
    draw_heatmap(i, data[i])
    

