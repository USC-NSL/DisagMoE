from benchmark.plotter.namer import add_args, get_plot_dir
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

for name in ["exp", "attn"]:
    fn = f"{args.path}/{name}_queueing_delay.csv"
    df = pd.read_csv(filepath_or_buffer=fn)

    worker_id = 0
    values = df.iloc[worker_id] * 1e3
    worker_name = f"{name}{worker_id}"

    plt.scatter(range(len(values)), values, marker='.')
    plt.xlabel('Steps')
    plt.ylabel('Queueing Delay (ms)')
    plt.title(f'Average Queueing Delay for {worker_name}')
    plt.savefig(f'{get_plot_dir(args.path)}/queueing_delay_steps.png')
    plt.close()
    
    sorted_values = sorted(values)
    cdf = [i / len(sorted_values) for i in range(len(sorted_values))]

    plt.plot(sorted_values, cdf)
    p90 = sorted_values[int(0.9 * len(sorted_values))]

    # plt.axhline(y=0.9, color='blue', linestyle='--', label='0.9')
    # plt.text(sorted_values[0], 0.9, '0.9', color='black', verticalalignment='bottom')

    plt.axvline(x=p90, color='blue', linestyle='dotted')

    plt.annotate(f'P90@{p90:.2f} ms', 
                 xy=(p90, 0), 
                 xytext=(p90, 0),
                 color='blue')
    
    p99 = sorted_values[int(0.99 * len(sorted_values))]

    plt.axvline(x=p99, color='red', linestyle='dotted')

    plt.annotate(f'P99@{p99:.2f} ms', 
                 xy=(p99, 0.5), 
                 xytext=(p99, 0.5),
                 color='red')
    
    plt.xlabel('Queueing Delay (ms)')
    plt.ylabel('CDF')
    plt.title(f'CDF of Queueing Delay for {worker_name}')
    plt.savefig(f'{get_plot_dir(args.path)}/queueing_delay_cdf.png')
    plt.close()