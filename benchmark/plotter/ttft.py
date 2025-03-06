from benchmark.plotter.namer import get_ttft_name
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--rate', type=float, required=True)
parser.add_argument('--num-nodes', type=int, default=1)

args = parser.parse_args()

fn = get_ttft_name(args)
df = pd.read_csv(fn)

df_sorted = df.sort_values(by=df.columns[0])
cdf = df_sorted[df.columns[0]].cumsum() / df_sorted[df.columns[0]].sum()

plt.figure()
plt.plot(df_sorted[df.columns[0]], cdf)
plt.xlabel('Time to First Token (s)')
plt.ylabel('CDF')
plt.title('CDF for TTFT')

plt.savefig(f"reports/throughput_benchmark/ttft/cdf_rate={args.rate}_nodes={args.num_nodes}.png")