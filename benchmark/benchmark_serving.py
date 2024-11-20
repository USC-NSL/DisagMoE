from disagmoe.frontend.controller import init_controller, Controller, AsyncResult
from disagmoe.utils.placement import ModelPlacement, ClusterConfig, get_model_placement
from disagmoe.utils.constants import *
from disagmoe.config import ModelConfig, CacheConfig, duo_expert_mixtral

import asyncio
from argparse import ArgumentParser

tokenizer = TOKENIZER_DEV_ID
sampler = SAMPLER_DEV_ID

master: Controller = None

def launch(args):
    cluster_config = ClusterConfig(n_node=1, n_gpu=3, 
                                id_tokenizer=tokenizer, 
                                id_sampler=sampler)

    model_config = duo_expert_mixtral
    model_config.num_layer = 16
    model_config.ep_size = 2
    model_config.num_experts = 8
    model_config.tp_size = 1

    mp = get_model_placement(model_config, cluster_config, "interleave")

    print(mp)
    
    global master

    master = init_controller(cluster_config.n_node, cluster_config.n_gpu)

    cache_config = CacheConfig(BLOCK_SIZE, 0.8, 2, "auto", 
                                num_gpu_blocks=NUM_BLOCKS, 
                                num_reserved_blocks=RESERVED_BLOCKS)

    master.init_engine(mp, model_config, cache_config, args.output_len)

    master.start_engine()
   
    
async def process_response(resp: AsyncResult):
    slo_stat = await resp.get()
    print(f">>> Response received: {resp.req_id}, {slo_stat}")

async def generate_requests(args):
    assert master is not None, "master is not initialized"
    print(f"generating requests at rate {args.rate} s/req")
    
    assert args.input_len == 1, "supports only 1 token as input"
    
    # start polling results
    master.start_polling_results()
    cnt = 0
    tasks = []
    if args.num_requests is None:
        while True:
            resp = master.put_single_request(args.input_len)
            asyncio.create_task(process_response(resp))
            await asyncio.sleep(args.rate)
    else:
        while cnt < args.num_requests:
            cnt += 1
            resp = master.put_single_request(args.input_len)
            task = asyncio.create_task(process_response(resp))
            tasks.append(task)
            await asyncio.sleep(args.rate)
        
        await asyncio.gather(*tasks)
        master.stop_workers()
    
def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("-r", "--rate", type=float, default=0.1, help="rate of incoming requests, seconds per request")
    parser.add_argument("-i", "--input-len", type=int, default=1, help="length of input sequence")
    parser.add_argument("-o", "--output-len", type=int, default=32, help="length of output sequence")
    parser.add_argument("-n", "--num-requests", type=int, default=None, help="number of requests to generate")
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    launch(args)
    
    asyncio.run(generate_requests(args))
    
if __name__ == "__main__":
    main()