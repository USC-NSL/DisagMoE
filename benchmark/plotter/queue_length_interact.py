from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import numpy as np

parser = ArgumentParser()
parser.add_argument('path', type=str, help="Path to the file or corresponding data directory")
parser.add_argument('--worker', type=int, required=True, help="Worker ID to plot")
parser.add_argument('--steps', type=int, default=2000, help="Number of steps to plot")
args = parser.parse_args()

data_path = f"{args.path}/queue_length.pkl" if not args.path.endswith('.pkl') else f"{args.path}"

with open(data_path, "rb") as f:
    data = pickle.load(f)
    
def sample_from_mid(ls, steps):
    if len(ls) <= steps:
        return ls
    start = (len(ls) - steps) // 2
    return ls[start:start+steps]

def draw_heatmap(data):
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
    
    initial_range = 200
    assert args.steps > initial_range, "steps should be greater than initial_range"
    
    step_ids = np.array(sample_from_mid(list(range(nsteps)), args.steps))
    step_start_time_ms_sampled = np.array(sample_from_mid(step_start_time_ms, args.steps))
    step_start_time_ms_sampled = step_start_time_ms_sampled - step_start_time_ms_sampled[0]
    step_start_time_ms_sampled = np.round(step_start_time_ms_sampled, 1)
    
    data = np.array([sample_from_mid(queue_length[layer_id], args.steps) for layer_id in layer_ids])
    
    plt.imshow(data, cmap='hot', origin='lower', extent=[0, args.steps, 0, nlayers])
    
    time_step_ms = 20
    # label xtick every time step
    total_time = step_start_time_ms_sampled[-1]
    per_step_time = total_time / args.steps
    xticks = np.arange(0, args.steps+1, time_step_ms/per_step_time)
    xtick_labels = np.arange(0, total_time+1, time_step_ms)
    num_xticks = min(len(xticks), len(xtick_labels))
    xticks = xticks[:num_xticks]
    xtick_labels = xtick_labels[:num_xticks]
    xtick_labels = [int(i) for i in xtick_labels]
    xticks = [int(i) for i in xticks]
    plt.xticks(xticks, xtick_labels)
    
    # plt.xticks(xticks, xtick_labels)
    
    # executed_layer_ids = np.argmax(data, axis=0)
    for i, layer_id in enumerate(sample_from_mid(step_executed_layer, args.steps)):
        plt.plot([i, i+1], [layer_id, layer_id], color='cyan', linestyle='--', linewidth=2)
    
    plt.xlim((args.steps - initial_range) / 2, (args.steps + initial_range) / 2)
    plt.ylim(0, nlayers)
    
    ax.set_aspect(2.5)
    plt.colorbar(label='Queue Length', orientation='vertical', shrink=0.5)
    plt.xlabel('time (ms)')
    plt.ylabel('layer')
    plt.title('queue length per layer')
    plt.tight_layout()
    plt.show()

# for each row of df, do draw_plot
# for i in range(len(data)):
#     draw_heatmap(i, data[i])

draw_heatmap(data[args.worker])
    

