import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

if __name__ == '__main__':
    # run the script
    import sys
    assert len(sys.argv) >= 2, "Usage: python latency_throughput.py <filename> [save_path]"
    filename = sys.argv[1]
    df = pd.read_csv(filename, sep=',')
    mean_itl = df['itl_latency_mean_ms']
    tput = df['token_throughput']
    
    x = tput
    y = mean_itl
    
    # Create a smooth curve through the points
    # Using spline interpolation for smoothness
    tck = interpolate.splrep(x, y, s=0)
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = interpolate.splev(x_smooth, tck, der=0)

    # Plot the smooth Pareto frontier
    plt.plot(x_smooth, y_smooth, 'r-', label='Pareto frontier')

    plt.plot(tput, mean_itl, 'o')
    plt.xlim(0, max(tput) * 1.1)
    plt.ylim(0, max(mean_itl) * 1.1)
    plt.ylabel('Mean ITL (ms)')
    plt.xlabel('Throughput (tokens/s)')
    plt.title('Throughput vs Mean ITL')
    save_path = filename.split('.')[0] + '_latency_throughput.png' if len(sys.argv) < 3 else sys.argv[2]
    plt.savefig(save_path, dpi=300)