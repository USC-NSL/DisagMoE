import asyncio
import time
import sys

from flask import Flask
from copy import copy

from benchmark.benchmark_serving import benchmark_serving, launch, benchmark_warmup
from benchmark.utils import get_parser_base

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
    rate = data.get('rate', 1.0)
    time = data.get('time', 10)
    distribution = data.get('distribution', 'poisson')
    
    if rate is None or time is None or distribution is None:
        return "Missing required parameters", 400
    
    try:
        rate = float(rate)
        time = int(time)
        distribution = str(distribution)
    except ValueError:
        return f"Invalid parameter types: {rate, time, distribution}", 400
    
    new_args = copy(args)
    new_args.rate = rate
    new_args.num_requests = int(time * rate)
    new_args.generator_type = distribution
    
    async def _runner():
        global master
        await master.start_scheduler()
        await benchmark_serving(master, new_args, is_api_server=True)
        await master.stop_scheduler()
    
    asyncio.run(_runner())
    
    return "run_once executed successfully", 200


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