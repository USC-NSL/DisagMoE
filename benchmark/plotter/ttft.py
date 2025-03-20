from benchmark.plotter.namer import add_args, get_plot_dir
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

fn = args.path
df = pd.read_csv(fn)

df_sorted = df.sort_values(by=df.columns[0])
cdf = df_sorted[df.columns[0]].cumsum() / df_sorted[df.columns[0]].sum()

plt.figure()
plt.plot(df_sorted[df.columns[0]], cdf)
plt.xlabel('Time to First Token (s)')
plt.ylabel('CDF')
plt.title('CDF for TTFT)')

plt.savefig(f"{get_plot_dir(args.path)}/ttft_cdf.png")