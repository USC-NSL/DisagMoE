from benchmark.plotter.namer import get_queue_length_name
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--rate', type=float, required=True)
parser.add_argument('--num-nodes', type=int, default=1)
parser.add_argument('--dp-size', type=int, default=1)
parser.add_argument('--ep-size', type=int, default=1)

report_dir = "reports/throughput_benchmark"

args = parser.parse_args()

df = pd.read_csv(get_queue_length_name(args))

def draw_plot(worker_id, queue_length):
    # enumerate queue_length as dict
    queue_length = dict(queue_length)
    plt.figure(figsize=(10, 5))
    
    for layer_id, length in queue_length.items():
        nsteps = len(length)
        plt.plot(list(range(nsteps)), length, '-', label=f'layer={layer_id}')
        
    plt.xlabel('step')
    plt.ylabel('queue length')
    plt.title(f'queue length per layer (Rate={args.rate}, Nodes={args.num_nodes})')
    plt.savefig(f'{report_dir}/queue_length_over_time/{args.rate}_nodes_{args.num_nodes}_dp-size={args.dp_size}_ep-size={args.ep_size}/{worker_id}.png')
    plt.close()

# for each row of df, do draw_plot
for i in range(len(df)):
    draw_plot(i, df.iloc[i])

