def get_sampler_step_name(args):
    return f"reports/sampler_step-rate={args.rate}-nodes={args.num_nodes}.csv"

def get_worker_queueing_delay_name(args, worker):
    return f"reports/{worker}_queueing_delay-rate={args.rate}-nodes={args.num_nodes}.csv"

def get_ttft_name(args):
    return f"reports/ttft-rate={args.rate}-nodes={args.num_nodes}.csv"