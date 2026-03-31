import threading
import time
import zmq
import importlib

ray = importlib.import_module("ray")

from typing import List, Dict, Set, Optional
from disagmoe.frontend.datatypes import SloStat, SamplerStepInfo, BatchDecodeResult, TokenizedRequest
from disagmoe.utils.logger import initialize_logger, get_logger
from disagmoe.utils.utils import get_ip

def t_now_high_ms() -> int:
    """Get current timestamp in milliseconds (equivalent to t_now_high in C++)"""
    return int(time.time() * 1000)

@ray.remote(num_cpus=2, num_gpus=0)
class Detokenizer:
    
    def __init__(self):
        self.finished_seqs: Set[int] = set()
        self.active_num_requests = 0
        self.slo_stats: Dict[int, SloStat] = {}
        self.step_infos: List[SamplerStepInfo] = []
        
        self.token_processed = 0
        self.iter = 0
        self.itl_buffer: list = []
        
        self.lock = threading.Lock()
        initialize_logger(f"Detokenizer")
        
        self.detokenizer_step_counter = 0
        
    def init_detokenizer_socket(self, detokenizer_port: str, host_ifname: str = "") -> str:
        context = zmq.Context(2)
        self.detokenizer_socket: zmq.Socket = context.socket(zmq.PULL)
        self.detokenizer_socket.bind(f"tcp://*:{detokenizer_port}")
        
        local_ip = get_ip(host_ifname)
        connect_addr = f"tcp://{local_ip}:{detokenizer_port}"
        
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        
        return connect_addr
    
    def run(self) -> None:
        self.last_log_ms = t_now_high_ms()
        while True:
            batch = self.detokenizer_socket.recv_pyobj()
            with self.lock:
                self.process_batch(batch)

    def process_batch(self, batch: BatchDecodeResult):
        num_tokens = len(batch.req_ids)
        cur_time_ms = t_now_high_ms()
        self.token_processed += num_tokens

        elapsed_ms = cur_time_ms - self.last_log_ms
        if elapsed_ms >= 10000:
            tput = self.token_processed * 1000 / elapsed_ms
            itl_str = ""
            if self.itl_buffer:
                self.itl_buffer.sort()
                n = len(self.itl_buffer)
                mean_itl = sum(self.itl_buffer) / n
                p50_itl = self.itl_buffer[n // 2]
                p99_itl = self.itl_buffer[int(n * 0.99)]
                itl_str = f" | ITL mean={mean_itl:.1f}ms p50={p50_itl:.1f}ms p99={p99_itl:.1f}ms"
            get_logger().info(
                f"Detokenizer: token throughput: {tput/1000:.2f}k tokens/s{itl_str}"
            )
            self.itl_buffer.clear()
            self.token_processed = 0
            self.last_log_ms = cur_time_ms
        
        for i in range(num_tokens):
            rid = batch.req_ids[i]
            
            if rid in self.slo_stats:
                stat = self.slo_stats[rid]
                if stat.t_tokens:
                    self.itl_buffer.append(cur_time_ms - stat.t_tokens[-1])
                stat.t_tokens.append(cur_time_ms)
            else:
                self.active_num_requests += 1
                self.slo_stats[rid] = SloStat(
                    req_id=rid,
                    t_prefill=cur_time_ms,
                    t_prefill_std=cur_time_ms,
                    t_decode=0.0,
                    t_tokens=[]
                )
        
        self.step_infos.append(SamplerStepInfo(
            num_tokens=num_tokens,
            time_stamp=cur_time_ms
        ))
        
        for i in range(num_tokens):
            rid = batch.req_ids[i]
            if batch.is_eos[i]:
                self.active_num_requests -= 1
                assert rid in self.slo_stats, f"Request {rid} not found in slo_stats"
                stat = self.slo_stats[rid]
                stat.t_decode = cur_time_ms
                stat.post_process()
                self.finished_seqs.add(rid)
        
        return num_tokens
    
    def fetch_finished_results(self) -> List[SloStat]:
        with self.lock:
            res = []
            for req_id in self.finished_seqs:
                res.append(self.slo_stats[req_id])
                del self.slo_stats[req_id]
            self.finished_seqs.clear()
            return res
    
    def fetch_sampler_step_infos(self) -> List[SamplerStepInfo]:
        with self.lock:
            infos = self.step_infos.copy()
            self.step_infos.clear()
            return infos
    
    def reset(self) -> None:
        with self.lock:
            self.step_infos.clear()
            self.slo_stats.clear()
            self.finished_seqs.clear()
            self.active_num_requests = 0
    
@ray.remote(num_cpus=2, num_gpus=0)
class Tokenizer:
    """
    Tokenizer for LLM serving.
    Tokenizes input text and sends it to the model.
    """
    
    def __init__(self, attn_dp_size: int):
        self.attn_dp_size = attn_dp_size
        self.worker_queues: List[zmq.Socket] = []
        self.t_submitted: Dict[int, float] = {}
            
    def init_tokenizer_sockets(self, tokenizer_ports: List[int], host_ifname: str = "") -> List[str]:
        local_ip = get_ip(host_ifname)
        context = zmq.Context(self.attn_dp_size)
        connect_addrs = []
        for i in range(self.attn_dp_size):
            addr = f"tcp://*:{tokenizer_ports[i]}"
            connect_addrs.append(f"tcp://{local_ip}:{tokenizer_ports[i]}")
            socket = context.socket(zmq.PUSH)
            socket.bind(addr)
            self.worker_queues.append(socket)
        return connect_addrs
    
    def put_single_request(
        self,
        req_id: int,
        init_prefill_len: int,
        max_output_len: int,
        dp_rank: int
    ) -> None:
        """
        Put a request into the tokenizer queue.
        
        Args:
            req_id: Request ID
            init_prefill_len: Initial prefill length
            max_output_len: Maximum output length
            tensor: Input tensor (should be 2D)
            dp_rank: Data parallel rank
        """
        tokenized_req = TokenizedRequest(req_id, init_prefill_len, max_output_len, [0])
        self.worker_queues[dp_rank].send_pyobj(tokenized_req)
        self.t_submitted[req_id] = time.time()
        
    def fetch_submitted_time(self) -> Dict[int, float]:
        return self.t_submitted
        
