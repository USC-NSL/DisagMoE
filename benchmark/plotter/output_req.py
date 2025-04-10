import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from benchmark.plotter.namer import add_args, get_plot_dir

parser = ArgumentParser()

parser = add_args(parser)

args = parser.parse_args()

fn = f"{args.path}/req_finish_time.csv"
df = pd.read_csv(fn)

df_sorted = df.sort_values(by=df.columns[0])

max_timestamp = max(df_sorted[df.columns[0]])

time_bins = range(0, int(max_timestamp), 2)
# give df a new column and fill all as 1
df_sorted['num_reqs'] = 1
time_sums = df_sorted.groupby(pd.cut(df_sorted[df.columns[0]], bins=time_bins))['num_reqs'].sum()
time_sums /= 2
plt.figure()
plt.plot(time_bins[:-1], time_sums, '-')
plt.xlabel('time (s)')
plt.ylabel('num of output request')
plt.title('output request per second')

plt.savefig(f"{get_plot_dir(args.path)}/output_reqs_over_time.png")