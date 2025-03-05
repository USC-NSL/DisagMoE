import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from benchmark.plotter.namer import get_sampler_step_name

parser = ArgumentParser()
parser.add_argument('--rate', type=float, required=True)
parser.add_argument('--gap-i', type=int)
parser.add_argument('--gap-t', type=int)
parser.add_argument('--seg', type=int)

CLK = 1e6

report_dir = "reports/throughput_benchmark"

args = parser.parse_args()

rate = args.rate
gap_i = args.seg or args.gap_i
gap_t = args.seg or args.gap_t

df = pd.read_csv(get_sampler_step_name(args))

_gap_i = (len(df.index) - gap_i + 1) // gap_i

# Summing up results in each gap for index
index_bins = range(0, len(df.index), _gap_i)
index_sums = df.groupby(pd.cut(df.index, bins=index_bins))['num_tokens'].sum()

plt.figure(figsize=(10, 5))
plt.plot(index_bins[:-1], index_sums, 'o-')
plt.xlabel('Steps')
plt.ylabel('Sum of Tokens')
plt.title('Sum of Tokens (By Steps)')
plt.savefig(f'{report_dir}/index/sum_sampler_step_rate_{rate}.png')
plt.close()

# Summing up results in each gap for time_stamp
df['time_stamp'] = (df['time_stamp'] - df['time_stamp'].iloc[0]) / CLK
_gap_t = (df['time_stamp'].iloc[-1] - df['time_stamp'].iloc[0]) / gap_t
time_bins = [
    df['time_stamp'].iloc[0] + i * _gap_t
        for i in range(gap_t + 1)
]
print(df['time_stamp'])
print(time_bins)
time_sums = df.groupby(pd.cut(df['time_stamp'], bins=time_bins))['num_tokens'].sum()

plt.figure(figsize=(10, 5))
plt.plot(time_bins[:-1], time_sums, 'o-')
plt.xlabel('Time (in seconds)')
plt.ylabel('Sum of Tokens')
plt.title('Sum of Tokens (By Time)')
plt.savefig(f'{report_dir}/time/sum_sampler_step_rate_{rate}.png')
plt.close()
