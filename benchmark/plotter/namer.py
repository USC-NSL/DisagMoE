import os
from argparse import ArgumentParser

def get_dir_path(args, trace_dir):
    dir_path = f"{trace_dir}/rate={args.rate}"
    # if dir not exists, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def get_sampler_step_name(args, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/sampler_step.csv"

def get_worker_queueing_delay_name(args, worker, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/{worker}_queueing_delay.csv"

def get_ttft_name(args, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/ttft.csv"

def get_req_submit_time_name(args, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/req_submit_time.csv"

def get_req_finish_time_name(args, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/req_finish_time.csv"

def get_trace_name(args, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/trace.json.gz"

def get_queue_length_name(args, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/queue_length.pkl"

def get_trace_metrics_name(args, trace_dir):
    dir_path = get_dir_path(args, trace_dir)
    return f"{dir_path}/trace_metrics.json"

def get_plot_dir(data_dir_path):
    plot_dir_path = f"{data_dir_path}/plots"
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path)
    return plot_dir_path

def add_args(parser: ArgumentParser):
    parser.add_argument('path', type=str, help="Directory where data is saved")
    return parser