import numpy as np
import pandas as pd

from typing import Dict, List
from queue import PriorityQueue
from dataclasses import dataclass

@dataclass
class Request:
    layer: int
    time: float
    n_tokens: int
    seq_lens: List[int]
    req_ids: List[int]
    
    def __lt__(self, other):
        return self.time < other.time
    
    
### Schedule methods

def schedule_largest_batch(waiting_queue: Dict[int, Request]):
    n_tokens = 0
    selected_layer = 0
    for layer, req in waiting_queue.items():
        if req.n_tokens > n_tokens:
            n_tokens = req.n_tokens
            selected_layer = layer
    return selected_layer

def schedule_fcfs(waiting_queue: Dict[int, Request]):
    n_time = 0
    selected_layer = 0
    for layer, req in waiting_queue.items():
        if req.time > n_time:
            n_time = req.time
            selected_layer = layer
    return selected_layer

def schedule_0(waiting_queue: Dict[int, Request]):
    return next(iter(waiting_queue.keys()))

def schedule_block(waiting_queue: Dict[int, Request]):
    pass

### initialization

pp_degree = 2

rate = 100
n_time = 30
n_request = n_time * rate
n_layers = 32 // pp_degree
output_len = 1024
max_bs = 256

np.random.seed(0)
gap = np.random.exponential(1 / rate, n_request)
arrivals = np.cumsum(gap)

t_calc = 3.0  # ms, per execution step
itl = 200  # ms, inter-layer latency

pending = PriorityQueue()
waiting_queue = {}  # Dict[layer_id -> Request]

for i, arrival in enumerate(arrivals):
    pending.put(Request(0, arrival * 1e3, 1, [0], [i]))  # seconds -> ms

### main loop

schedule = schedule_largest_batch
# schedule = schedule_fcfs
# schedule = schedule_0

records = {}  # Dict[req_id -> List[TimeStamp]]

cur = 0
while not pending.empty() or len(waiting_queue) > 0:
    cur += t_calc
    
    # put the new requests into the waiting queue
    while not pending.empty():
        req: Request = pending.get()
        if req.time > cur:
            pending.put(req)
            break
        if req.layer not in waiting_queue:
            waiting_queue[req.layer] = req
        else:
            waiting_queue[req.layer].n_tokens += req.n_tokens
            waiting_queue[req.layer].seq_lens += req.seq_lens
            waiting_queue[req.layer].req_ids += req.req_ids
    
    if len(waiting_queue) == 0:
        continue
    
    # process the requests
    layer = schedule(waiting_queue)
    req = waiting_queue[layer]
    # print(req.n_tokens)
    if req.n_tokens <= max_bs:
        del waiting_queue[layer]
    else:
        new_req = Request(layer, cur, max_bs, req.seq_lens[:max_bs], req.req_ids[:max_bs])
        req.n_tokens -= max_bs
        req.seq_lens = req.seq_lens[max_bs:]
        req.req_ids = req.req_ids[max_bs:]
        req = new_req
    req.layer = (req.layer + 1) % n_layers
    removed_idx = []
    for i in range(len(req.seq_lens)):
        records[req.req_ids[i]] = records.get(req.req_ids[i], []) + [cur]
        req.seq_lens[i] += 1
        if req.seq_lens[i] == output_len:
            removed_idx.append(i)
    for idx in sorted(removed_idx, reverse=True):
        del req.seq_lens[idx]
    req.n_tokens -= len(removed_idx)
    req.time += itl
    if req.n_tokens > 0:
        pending.put(req)

print("Finalization time:", cur, "ms")

lats = []
for _, timestamps in records.items():
    for i in range(1, len(timestamps)):
        lats.append(timestamps[i] - timestamps[i - 1])
lat_med = np.median(lats)
lat_p90 = np.percentile(lats, 90)

print("Schedule:", schedule.__name__)
print("Mean ITL:", np.mean(lats), "ms")
print("Median ITL:", lat_med, "ms")
print("P90 ITL:", lat_p90, "ms")