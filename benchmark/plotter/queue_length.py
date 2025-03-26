from benchmark.plotter.namer import add_args, get_plot_dir
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

parser = ArgumentParser()
parser = add_args(parser)
parser.add_argument('--steps', type=int, default=200)
args = parser.parse_args()

data_path = f"{args.path}/queue_length.pkl"

plot_dir = f"{get_plot_dir(args.path)}/queue_length_over_time/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

with open(data_path, "rb") as f:
    data = pickle.load(f)
    
def sample_from_mid(ls, steps):
    if len(ls) <= steps:
        return ls
    start = (len(ls) - steps) // 2
    return ls[start:start+steps]

def draw_heatmap(worker_id, data):
    # enumerate queue_length as dict
    queue_length, step_executed_layer, step_start_time_ms = data
    
    plt.figure(figsize=(20, 8))
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
    plt.xticks(np.arange(args.steps)[-1 : ], step_start_time_ms_sampled[-1 : ], rotation=0)
    
    # executed_layer_ids = np.argmax(data, axis=0)
    for i, layer_id in enumerate(sample_from_mid(step_executed_layer, args.steps)):
        plt.plot([i, i+1], [layer_id, layer_id], color='cyan', linestyle='--', linewidth=2)
    
    ax.set_aspect(2.5)
    plt.xlabel('time (ms)')
    plt.ylabel('layer')
    plt.title('queue length per layer')
    plt.colorbar(label='Queue Length', orientation='vertical', shrink=0.5)
    plt.savefig(f'{plot_dir}/{worker_id}.png', bbox_inches='tight', dpi=300)
    plt.close()

# for each row of df, do draw_plot
for i in range(len(data)):
    draw_heatmap(i, data[i])
    

