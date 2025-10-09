import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from benchmark.plotter.namer import add_args, get_plot_dir

parser = ArgumentParser()

parser = add_args(parser)

args = parser.parse_args()

fn_finish = f"{args.path}/req_finish_time.csv"
fn_submit = f"{args.path}/req_submit_time.csv"

# Function to process dataframe and get requests per second
def process_requests(df, bin_size=2):
    df_sorted = df.sort_values(by=df.columns[0])
    max_timestamp = max(df_sorted[df.columns[0]])
    time_bins = range(0, int(max_timestamp) + bin_size, bin_size)
    # give df a new column and fill all as 1
    df_sorted['num_reqs'] = 1
    time_sums = df_sorted.groupby(pd.cut(df_sorted[df.columns[0]], bins=time_bins))['num_reqs'].sum()
    time_sums /= bin_size
    return time_bins[:-1], time_sums

# Read submit data (input requests)
df_submit = pd.read_csv(fn_submit)
time_bins_submit, reqs_submit = process_requests(df_submit)

# Read finish data (output requests)
df_finish = pd.read_csv(fn_finish)
time_bins_finish, reqs_finish = process_requests(df_finish)

# Create a single plot with both request types
plt.figure(figsize=(10, 6))
plt.plot(time_bins_submit, reqs_submit, '-b', label='Input Requests')
plt.plot(time_bins_finish, reqs_finish, '-r', label='Output Requests')
plt.xlabel('time (s)')
plt.ylabel('requests per second')
plt.title('Input and Output Requests per Second')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f"{get_plot_dir(args.path)}/requests_over_time.png")