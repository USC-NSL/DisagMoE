from benchmark.plotter.namer import add_args, get_plot_dir
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

parser = ArgumentParser()
parser = add_args(parser)
parser.add_argument('--steps', type=int, default=200)
parser.add_argument('-t', '--time-unit', default=0.2, type=float, help='time step in ms')
parser.add_argument('-l', '--layers', default=32, type=int, help='number of layers')
args = parser.parse_args()

data_path = f"{args.path}/queue_length.pkl"

plot_dir = f"{get_plot_dir(args.path)}/queue_length_over_wall_time/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

with open(data_path, "rb") as f:
    data = pickle.load(f)
    
def sample(ls, steps):
    starting_percentile = 0.8
    if len(ls) <= steps:
        return ls
    start = int(len(ls) * starting_percentile)
    end = start + steps
    return ls[start:end]

def step_to_timestamp(q, t):
    # q: [step, layer]
    
    nsteps = len(q)
    
    start_time_ms = t[0]
    end_time_ms = t[-1]
    
    queue_length_over_time = []
    base_t = start_time_ms
    cur_step = 0
    while base_t < end_time_ms:
        while cur_step < nsteps - 1 and base_t > t[cur_step + 1]:
            cur_step += 1
        queue_length_over_time.append(q[cur_step])
        base_t += args.time_unit
    
    return np.array(queue_length_over_time)

def draw_heatmap(worker_id, data):
    # enumerate queue_length as dict
    queue_length, step_executed_layer, step_start_time_ms = data
    
    plt.figure(figsize=(20, 12))
    figure, ax = plt.subplots()
    
    layer_ids = []
    nsteps = 0
    
    for layer_id, length in queue_length.items():
        layer_ids.append(layer_id)
        nsteps = len(length)
        
    layer_ids.sort()
    nrows = len(layer_ids)
    nlayers = args.layers
    ngroups = nrows // nlayers
    
    step_ids = np.array(sample(list(range(nsteps)), args.steps))
    step_start_time_ms_sampled = np.array(sample(step_start_time_ms, args.steps))
    step_start_time_ms = step_start_time_ms_sampled - step_start_time_ms_sampled[0]
    
    # [layer, step]
    queue_length_over_step = np.array([sample(queue_length[layer_id], args.steps) for layer_id in layer_ids])
    executed_layer_over_step = np.array(sample(step_executed_layer, args.steps))
    queue_length_over_time = step_to_timestamp(queue_length_over_step.transpose(), step_start_time_ms).transpose()
    executed_layer_over_time = step_to_timestamp(executed_layer_over_step, step_start_time_ms)
    
    total_time = step_start_time_ms[-1]
    x_lim = int(total_time / args.time_unit)
    print(f"worker {worker_id} total time: {total_time} ms")
    print(f"queue length shape {queue_length_over_time.shape}")
    print(f"layer_ids {layer_ids}")
    
    plt.imshow(queue_length_over_time, cmap='hot', origin='lower', extent=[0, x_lim, 0, nrows])
    
    if ngroups > 1:
        for i in range(ngroups, nrows, ngroups):
            plt.axhline(y=i, color='white', linestyle='-', linewidth=0.5)
        
    # label xtick every time step
    xtick_time_step = 20 # ms 
    xticks = np.arange(0, x_lim, int(xtick_time_step / args.time_unit))
    xtick_labels = np.arange(0, int(total_time), xtick_time_step)
    num_xticks = min(len(xticks), len(xtick_labels))
    xticks = xticks[:num_xticks]
    xtick_labels = xtick_labels[:num_xticks]
    xtick_labels = [int(i) for i in xtick_labels]
    xticks = [int(i) for i in xticks]
    plt.xticks(xticks, xtick_labels)
    plt.yticks(np.arange(ngroups / 2, nrows, ngroups), np.arange(nlayers), fontsize=6)
    
    for i, layer_id in enumerate(executed_layer_over_time):
        if executed_layer_over_time[i] == -1:
            continue
        plt.plot([i, i+1], [layer_id, layer_id], color='cyan', linestyle='--', linewidth=1)
        
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, nrows)
    ax.set_aspect(15)
    
    plt.xlabel('time (ms)')
    plt.ylabel('layer')
    plt.title('queue length per layer')
    plt.colorbar(label='Queue Length', orientation='vertical', shrink=0.5)
    plt.savefig(f'{plot_dir}/{worker_id}.png', bbox_inches='tight', dpi=900)
    plt.close()

# for each row of df, do draw_plot
for i in range(len(data)):
    draw_heatmap(i, data[i])


