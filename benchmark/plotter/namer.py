import os
from argparse import ArgumentParser

def get_dir_path(args):
    dir_path = f"reports/rate={args.rate}-nodes={args.num_nodes}-dp={args.dp_size}-ep={args.ep_size}-"\
               f"layersch={args.layer_scheduler_type}-layerstep={args.layer_scheduler_step}"
    # if dir not exists, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def get_sampler_step_name(args):
    dir_path = get_dir_path(args)
    return f"{dir_path}/sampler_step.csv"

def get_worker_queueing_delay_name(args, worker):
    dir_path = get_dir_path(args)
    return f"{dir_path}/{worker}_queueing_delay.csv"

def get_ttft_name(args):
    dir_path = get_dir_path(args)
    return f"{dir_path}/ttft.csv"

def get_req_finish_time_name(args):
    dir_path = get_dir_path(args)
    return f"{dir_path}/req_finish_time.csv"

def get_trace_name(args):
    dir_path = get_dir_path(args)
    return f"{dir_path}/trace.json.gz"

def get_queue_length_name(args):
    dir_path = get_dir_path(args)
    return f"{dir_path}/queue_length.pkl"

def get_trace_metrics_name(args):
    dir_path = get_dir_path(args)
    return f"{dir_path}/trace_metrics.json"

def get_plot_dir(data_dir_path):
    plot_dir_path = f"{data_dir_path}/plots"
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path)
    return plot_dir_path

def add_args(parser: ArgumentParser):
    parser.add_argument('path', type=str, help="Directory where data is saved")
    return parser