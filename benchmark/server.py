import asyncio
import time
import sys

from flask import Flask
from copy import copy

from benchmark.benchmark_serving import benchmark_serving, launch, benchmark_warmup
from benchmark.utils import get_parser_base
from benchmark.workload import get_generator

from disagmoe.utils.logger import get_logger
from disagmoe.frontend.controller import Controller


logger = get_logger("BenchmarkServer")

app = Flask(__name__)
master: Controller = None
args = None

@app.route('/run_once', methods=['POST'])
def run_once_endpoint():
    from flask import request
    
    data = request.get_json()
    rate = data.get('rate', 10)
    duration = data.get('time', 10)
    distribution = data.get('distribution', 'poisson')
    min_input_len = data.get('min_input_len', 30)
    max_input_len = data.get('max_input_len', 70)
    min_output_len = data.get('min_output_len', 80)
    max_output_len = data.get('max_output_len', 120)
    
    if rate is None or duration is None or distribution is None:
        return "Missing required parameters", 400
    
    try:
        rate = int(rate)
        duration = int(duration)
        distribution = str(distribution)
    except ValueError:
        return f"Invalid parameter types: {rate, duration, distribution}", 400
    
    new_args = copy(args)
    new_args.rate = rate
    new_args.generator_type = distribution
    generator_type = get_generator(distribution)
    generator = generator_type(rate, 0, 0, 0, 0, 0)
    new_args.num_requests = generator.get_num_requests(duration)
    new_args.min_input_len = min_input_len
    new_args.max_input_len = max_input_len
    new_args.min_output_len = min_output_len
    new_args.max_output_len = max_output_len
    print(f"put {new_args.num_requests} requests")
    
    async def _runner():
        global master
        master.start_polling_results()
        await master.start_scheduler()
        metrics = await benchmark_serving(master, new_args, is_api_server=True)
        print("Metrics:", metrics)
        await master.stop_scheduler()
        return metrics
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    metrics = loop.run_until_complete(_runner())
    
    return f"run_once executed successfully\n{metrics}\n", 200


@app.route('/set_schedule', methods=['POST'])
def set_schedule_endpoint():
    global master, args
    from flask import request
    
    data = request.get_json()
    policy = data.get('policy')
    step = data.get('step')
    
    if policy is None and step is None:
        return "Missing required parameters: policy or step", 400
    
    if policy is not None:
        try:
            policy = str(policy)
        except ValueError:
            return f"Invalid parameter type: {policy}", 400
        
        if policy not in ["mbfs", "flfs", "mbflfs"]:
            return f"Invalid policy: {policy}", 400
        
        master.set_schedule_policy(policy)
        args.layer_scheduler_type = policy
    
    if step is not None:
        try:
            step = int(step)
        except ValueError:
            return f"Invalid parameter type: {step}", 400
        
        master.set_schedule_block(step)
        args.layer_scheduler_step = step
    
    return "set_schedule executed successfully", 200


async def init(master: Controller, args):
    master.start_polling_results()
    await master.start_scheduler()
    await benchmark_warmup(master, args)
    await master.stop_scheduler()

def main():
    global master, args
    parser = get_parser_base()
    args = parser.parse_args()
    
    logger.info("Launching DisagMoE Controller")
    
    master = launch(args)
    asyncio.run(init(master, args))
    
    logger.info("DisagMoE Controller launched.")
    
    logger.info("Launching Flask Server")
    time.sleep(2)
    sys.stdout.flush()
    sys.stderr.flush()
    
    app.run(host='0.0.0.0', port=6699)
    

if __name__ == '__main__':
    main()