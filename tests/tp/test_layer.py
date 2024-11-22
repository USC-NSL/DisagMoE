import os
import ray
import torch
import torch.distributed as dist

from disagmoe.config import ModelConfig, CacheConfig, mixtral_config
from disagmoe.models.distributed import set_tensor_model_parallel_config, set_linear_method_init_value
from disagmoe.executor.executor import ParallelAttnExecutor, AttnExecutor
from disagmoe.frontend.engine import FlashAttentionMetadata
from torch.nn.utils.rnn import pad_sequence

import time

DEFAULT_VALUE = 0.05

@ray.remote(num_gpus=1)
class Worker:
    
    def __init__(self, device_id, model_config: ModelConfig, cache_config: CacheConfig):
        print(model_config)
        self.device_id = device_id
        self.model_config = model_config
        self.cache_config = cache_config
        
    def setup(self):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        set_linear_method_init_value(DEFAULT_VALUE)
        set_tensor_model_parallel_config(self.model_config)
        if self.model_config.tp_size > 1:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "26500"
            dist.init_process_group(backend="nccl", world_size=self.model_config.tp_size, rank=self.model_config.rank, init_method="env://")
            dist.barrier()
            self.nccl_barrier()
            self.executor = ParallelAttnExecutor(self.model_config, self.cache_config)
        else:
            self.executor = AttnExecutor(self.model_config, self.cache_config)
    
    def nccl_barrier(self):
        tmp = torch.zeros((2048, )).to("cuda")
        dist.broadcast(tmp, src=0)
        torch.cuda.synchronize()
    
    def execute(self, tensor: torch.Tensor, meta: FlashAttentionMetadata):
        
        profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name="./reports", 
                    worker_name=f"worker-{self.device_id}",
                    use_gzip=True,))
        profiler.start()
        
        positions = torch.zeros(tensor.shape[0], dtype=torch.long).to("cuda", non_blocking=True)
        hiddens, expert_ids = self.executor.execute(0, positions, tensor, meta)
        
        torch.cuda.synchronize()
        time.sleep(1)
        
        st = time.time()
        hiddens, expert_ids = self.executor.execute(0, positions, tensor, meta)
        torch.cuda.synchronize()
        if self.model_config.rank == 0:
            print("Time taken:", time.time() - st)
        
        profiler.stop()
        
        return hiddens, expert_ids
    
    def sync(self):
        pass

block_size = 32

def make_seqlens(lens):
    seqlen = [0]
    for l in lens:
        seqlen.append(seqlen[-1] + l)
    return torch.tensor(seqlen, dtype=torch.int32, device=torch.get_default_device())

def make_naive_mapping(lens, mode):
    block_table = []
    slots_table = []
    allocated_blocks = 4
    for l in lens:
        num_blocks = (l + block_size) // block_size
        start = allocated_blocks
        end = num_blocks + allocated_blocks
        block_list = list(range(start, end))
        allocated_blocks = end
        block_table.append(torch.tensor(block_list, dtype=torch.int32))
        if mode == "prefill":
            start_slot = start * block_size
            end_slot = start_slot + l
            slots_list = list(range(start_slot, end_slot))
            slots_table.extend(slots_list)
        elif mode == "decode":
            end_slot = start * block_size + l - 1
            slots_table.append(end_slot)
        else:
            assert False
            
    block_table = pad_sequence(block_table, batch_first=True, padding_value=0)
    slots_table = torch.tensor(slots_table, dtype=torch.long)
    return block_table, slots_table

def make_prefill_meta(num_prefills: int):
    lens = [1 for _ in range(num_prefills)]
    seqlens = torch.tensor(lens)
    num_prefill_tokens = sum(lens)
    seqlens = torch.tensor(lens, dtype=torch.int32, device=torch.get_default_device())
    seqlens_q = make_seqlens(lens)
    context_lens_tensor = [0] * num_prefills
    seqlens_kv = seqlens_q
    max_seqlen_q = max(lens)
    max_seqlen_kv = max_seqlen_q
    block_table, slot_mapping = make_naive_mapping(lens, "prefill")
    meta = FlashAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=0,
        slot_mapping=slot_mapping,
        seq_lens=lens,
        seq_lens_tensor=seqlens,
        max_query_len=max_seqlen_q,
        max_prefill_seq_len=max_seqlen_q,
        max_decode_seq_len=0,
        query_start_loc=seqlens_q,
        seq_start_loc=seqlens_kv,
        context_lens_tensor=context_lens_tensor,
        block_tables=torch.tensor([]),
        use_cuda_graph=False,
    )
    return meta

def main():
    set_linear_method_init_value(DEFAULT_VALUE)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    tasks = []
    
    # TP = 1
    model_config = mixtral_config
    model_config.tp_size = 1
    model_config.ep_size = 1
    model_config.tp_enable_inter_group = False
    model_config.num_layers = 1
    cache_config = cache_config = CacheConfig(
        block_size=32,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        num_gpu_blocks=4096,
        num_reserved_blocks=1024,
    )
    worker = Worker.remote(0, model_config, cache_config)
    worker.setup.remote()
    bs = 256
    meta = make_prefill_meta(num_prefills=bs)
    # torch.manual_seed(123)
    tensor = torch.randn(bs, model_config.hidden_size).to("cuda")
    std, _ = ray.get(worker.execute.remote(tensor, meta))
    print(std)
    del worker
    
    # TP > 1
    n = 4
    model_config.tp_size = n
    workers = []
    for i in range(n):
        model_config.rank = i
        workers.append(Worker.remote(i + 1, model_config, cache_config))
        workers[-1].setup.remote()
    
    ray.get([w.sync.remote() for w in workers])
    print("inited")
    
    for i in range(n):
        tasks.append(workers[i].execute.remote(tensor, meta))
    
    results = ray.get(tasks)
    out: torch.Tensor = results[0][0]
    print(out)
    print(((out - std) / std))
    print(((out - std) / std).abs().max())
    assert ((out - std) / std).abs().max() < 1e-6
    
main()