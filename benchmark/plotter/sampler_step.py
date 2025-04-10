import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from benchmark.plotter.namer import add_args, get_plot_dir

parser = ArgumentParser()
parser = add_args(parser)
parser.add_argument('--gap-i', type=int, default=1)
parser.add_argument('--gap-t', type=int, default=5)

CLK = 1e6

args = parser.parse_args()

fn = f"{args.path}/sampler_step.csv"
df = pd.read_csv(fn)

# Summing up results in each gap for index

gap_i = args.gap_i

index_bins = range(0, len(df.index), gap_i)
index_sums = df.groupby(pd.cut(df.index, bins=index_bins))['num_tokens'].sum()

plt.figure(figsize=(10, 5))
plt.plot(index_bins[:-1], index_sums, '-')
plt.xlabel('Steps')
plt.ylabel(f'Number of Tokens per {gap_i} steps')
plt.title('Sampler\'s Average Output Tokens)')
plt.savefig(f'{get_plot_dir(args.path)}/output_tokens_over_index.png')
plt.close()

# Summing up results in each gap for time_stamp

df['time_stamp'] = (df['time_stamp'] - df['time_stamp'].iloc[0]) / CLK
gap_t = args.gap_t
seg = int((df['time_stamp'].iloc[-1] - df['time_stamp'].iloc[0] + gap_t - 1) // gap_t)
time_bins = [
    df['time_stamp'].iloc[0] + i * gap_t
        for i in range(seg + 1)
]
# print(df['time_stamp'])
# print(time_bins)
time_sums = df.groupby(pd.cut(df['time_stamp'], bins=time_bins))['num_tokens'].sum()
time_sums /= gap_t
print(f"peak throughput: {time_sums.max()} tokens/s")
plt.figure(figsize=(10, 5))
plt.plot(time_bins[:-1], time_sums, '-')
plt.axvline(x=120, color='green', linestyle='dotted')
plt.xlabel('Time (in seconds)')
plt.ylabel('Number of Tokens per second')
plt.title('Sampler\'s Average Output Tokens)')
plt.savefig(f'{get_plot_dir(args.path)}/output_tokens_over_time.png')
plt.close()
