import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from benchmark.plotter.namer import get_req_finish_time_name

parser = ArgumentParser()
parser.add_argument('--rate', type=float, required=True)
parser.add_argument('--num-nodes', type=int, default=1)
parser.add_argument('--dp-size', type=int, default=1)
parser.add_argument('--ep-size', type=int, default=1)

report_dir = "reports/throughput_benchmark"

args = parser.parse_args()

fn = get_req_finish_time_name(args)
df = pd.read_csv(fn)

df_sorted = df.sort_values(by=df.columns[0])

max_timestamp = max(df_sorted[df.columns[0]])

time_bins = range(0, int(max_timestamp))
time_sums = df_sorted.groupby(pd.cut(df_sorted[df.columns[0]], bins=time_bins))['num_tokens'].sum()

plt.figure()
plt.plot(time_bins, time_sums, '-')
plt.xlabel('time (s)')
plt.ylabel('num of output request')
plt.title(f'output request per second(rate={args.rate}, nodes={args.num_nodes})')

plt.savefig(f"{report_dir}/output_reqs_over_time/req_throughput_rate={args.rate}_nodes={args.num_nodes}.png")