import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from benchmark.plotter.namer import get_sampler_step_name, get_plot_dir

parser = ArgumentParser()
parser.add_argument('--rate', type=float, required=True)
parser.add_argument('--gap-i', type=int, default=1)
parser.add_argument('--gap-t', type=int, default=1)
parser.add_argument('--seg', type=int)
parser.add_argument('--num-nodes', type=int, default=1)
parser.add_argument('--dp-size', type=int, default=1)
parser.add_argument('--ep-size', type=int, default=1)

CLK = 1e6

args = parser.parse_args()

rate = args.rate
num_nodes = args.num_nodes

assert args.seg or (args.gap_i and args.gap_t)

df = pd.read_csv(get_sampler_step_name(args))

# Summing up results in each gap for index

gap_i = (len(df.index) - args.seg + 1) // args.seg if args.seg else args.gap_i

index_bins = range(0, len(df.index), gap_i)
index_sums = df.groupby(pd.cut(df.index, bins=index_bins))['num_tokens'].sum()

plt.figure(figsize=(10, 5))
plt.plot(index_bins[:-1], index_sums, '-')
plt.xlabel('Steps')
plt.ylabel(f'Number of Tokens per {gap_i} steps')
plt.title(f'Sampler\'s Average Output Tokens (Rate={rate}, Nodes={num_nodes})')
plt.savefig(f'{get_plot_dir()}/output_tokens_over_index/token_throughput_rate_{rate}_nodes_{num_nodes}_dp-size={args.dp_size}_ep-size={args.ep_size}.png')
plt.close()

# Summing up results in each gap for time_stamp

df['time_stamp'] = (df['time_stamp'] - df['time_stamp'].iloc[0]) / CLK
gap_t = (df['time_stamp'].iloc[-1] - df['time_stamp'].iloc[0]) / args.seg if args.seg else args.gap_t
seg = args.seg if args.seg else int((df['time_stamp'].iloc[-1] - df['time_stamp'].iloc[0] + gap_t - 1) // gap_t)
time_bins = [
    df['time_stamp'].iloc[0] + i * gap_t
        for i in range(seg + 1)
]
print(df['time_stamp'])
print(time_bins)
time_sums = df.groupby(pd.cut(df['time_stamp'], bins=time_bins))['num_tokens'].sum()

plt.figure(figsize=(10, 5))
plt.plot(time_bins[:-1], time_sums, '-')
plt.axvline(x=120, color='green', linestyle='dotted')
plt.xlabel('Time (in seconds)')
plt.ylabel('Number of Tokens per second')
plt.title(f'Sampler\'s Average Output Tokens (Rate={rate}, Nodes={num_nodes})')
plt.savefig(f'{get_plot_dir()}/output_tokens_over_time/token_throughput_rate_{rate}_nodes_{num_nodes}_dp-size={args.dp_size}_ep-size={args.ep_size}.png')
plt.close()
