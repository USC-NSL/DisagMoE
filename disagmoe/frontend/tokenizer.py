import threading
import time
import torch
import zmq
import pickle

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from disagmoe.frontend.datatypes import SloStat, SamplerStepInfo

def t_now_high_ms() -> int:
    """Get current timestamp in milliseconds (equivalent to t_now_high in C++)"""
    return int(time.time() * 1000)
    
@dataclass
class BatchDecodeResult:
    
    req_ids: List[int]
    token_ids: List[int]
    is_eos: List[bool]

class Detokenizer:
    
    def __init__(self, recv_addr: str):
        self.finished_seqs: Set[int] = set()
        self.active_num_requests = 0
        self.slo_stats: Dict[int, SloStat] = {}
        self.step_infos: List[SamplerStepInfo] = []
        
        self.result_queue: zmq.Socket = zmq.Socket(zmq.PULL)
        self.result_queue.bind(recv_addr)
        
        self.lock = threading.Lock()
        self.working_thread = threading.Thread(target=self.run)
        self.working_thread.start()
        
        self.token_processed = 0
        self.iter = 0
        self.start_timestamp_ms = t_now_high_ms()
        
    def run(self) -> None:
        while True:
            batch = self.result_queue.recv_pyobj()
            
            with self.lock:
                self.process_batch(batch)
                
            num_tokens = len(batch.req_ids)
            self.token_processed += num_tokens
            self.iter += 1
            
            if self.iter >= 500:
                self.log_throughput()
                
    def log_throughput(self) -> None:
        cur_time_ms = t_now_high_ms()
        elapsed_time_ms = cur_time_ms - self.start_timestamp_ms
        token_throughput = self.token_processed * 1000 / elapsed_time_ms
        print(f"Detokenizer: token throughput: {token_throughput} tokens/s")
        self.token_processed = 0
        self.iter = 0
        self.start_timestamp_ms = cur_time_ms

    def process_batch(self, batch: BatchDecodeResult):
        num_tokens = len(batch.req_ids)
        cur_time_ms = t_now_high_ms()
        
        for i in range(num_tokens):
            rid = batch.req_ids[i]
            
            if rid in self.slo_stats:
                # Store timestamp in milliseconds (matching C++ behavior)
                self.slo_stats[rid].t_tokens.append(cur_time_ms)
            else:
                # Initialize SLO stat for new request
                # Store times in milliseconds initially (matching C++), convert to seconds for SloStat
                self.active_num_requests += 1
                self.slo_stats[rid] = SloStat(
                    req_id=rid,
                    t_prefill=cur_time_ms / 1000.0,  # Convert to seconds
                    t_prefill_std=cur_time_ms / 1000.0,  # Convert to seconds
                    t_decode=0.0,
                    t_tokens=[cur_time_ms]  # Store in milliseconds initially
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
                self.slo_stats[rid].t_decode = cur_time_ms / 1000.0  # Convert to seconds
                self.finished_seqs.add(rid)
        
        return num_tokens
    
    def fetch_finished_slo_stats(self) -> List[SloStat]:
        with self.lock:
            res = []
            for req_id in self.finished_seqs:
                res.append(self.slo_stats[req_id])
                del self.slo_stats[req_id]
            self.finished_seqs.clear()
            return res
    
    def fetch_step_infos(self) -> List[SamplerStepInfo]:
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

@dataclass
class TokenizedRequest:
    
    req_id: int
    init_prefill_len: int
    max_output_len: int
    token_ids: List[int]
    
class Tokenizer:
    """
    Tokenizer for LLM serving.
    Tokenizes input text and sends it to the model.
    """
    
    def __init__(self, attn_dp_size: int, attn_engine_addrs: List[str]):
        self.attn_dp_size = attn_dp_size
        self.attn_engine_addrs = attn_engine_addrs
        self.worker_queues: List[zmq.Socket] = []
        for addr in attn_engine_addrs:
            ctx = zmq.Context()
            socket = ctx.socket(zmq.PUSH)
            socket.connect(addr)
            self.worker_queues.append(socket)
    
    def put_request(
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