from matplotlib import pyplot as plt
import pandas as pd

def plot_res(sgl_res, amoe_res, type):

    mx = 0
    my = 0
    
    tput = sgl_res['token_tput']
    mean_itl = sgl_res['mean_latency (ms)']
    
    mx = max(tput)
    my = max(mean_itl)
    
    plt.figure(figsize=(8, 6))
    
    # set font size to 16
    plt.rcParams.update({'font.size': 16})
    
    # SGLang
    plt.plot(tput, mean_itl, 'o', label='SGLang', color='blue')
    plt.plot(tput, mean_itl, '-', alpha=0.5, color='blue')

    # AsyncMoE
    tput = amoe_res['token_throughput']
    mean_itl = amoe_res['itl_latency_mean_ms']
    plt.plot(tput, mean_itl, 'o', label='AsyncMoE', color='red')
    plt.plot(tput, mean_itl, '-', alpha=0.5, color='red')
    
    mx = max(max(tput), mx)
    my = max(max(mean_itl), my)
    
    plt.legend(loc="upper left")
    plt.xlim(0, mx * 1.1)
    plt.ylim(0, my * 1.1)
    plt.ylabel('Mean ITL (ms)')
    plt.xlabel('Throughput (tokens/s)')
    plt.title('Throughput vs Mean ITL')
    plt.tight_layout()
    plt.savefig(f"{type}.png", dpi=300)


def plot_reasonable_v2_mqa_top1(lens_type, model_type):
    sglang_df = pd.read_csv(f"/mnt/efs/baseline/scale_metrics/resonable_v2_top1_mqa/result.csv")
    amoe_df = pd.read_csv(f"./mqa_top1/benchmark.csv")
    plot_res(sglang_df, amoe_df, "reasonable_v2_" + model_type)

plot_reasonable_v2_mqa_top1("reasonable_v2", "mqa_top1")


# def plot(lens_type, model_type):
#     sglang_df = pd.read_csv(f"/mnt/efs/baseline/scale_metrics/top1_gqa/reasonable/result.csv")
#     amoe_df = pd.read_csv(f"./gqa_top1/benchmark.csv")
#     plot_res(sglang_df, amoe_df, "reasonable_" + model_type)

# plot("reasonable", "gqa_top1")