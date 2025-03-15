import os

def get_dir_path(args):
    dir_path = f"reports/rate={args.rate}-nodes={args.num_nodes}-dp={args.dp_size}-ep={args.ep_size}"
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