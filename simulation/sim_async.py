import simpy
import random
import os
from collections import deque
from dataclasses import dataclass
from util import expert_compute_time_lookup_table_from_profile


# ------------------------------
# Configurable parameters
# ------------------------------

EP_GROUP_SIZE      = 4      # "n": number of expert workers per layer
TOTAL_EXPERT_COUNT = 16     # total experts per layer
MAX_BATCH_SIZE     = 512
NUM_LAYERS         = 48      # number of expert layers
TOTAL_TOKENS       = 10_000

ARRIVAL_RATE       = 50.0   # lambda for Poisson arrivals (tokens / tick)

ATTN_SERVICE_T     = 1.0    # time (ticks) per token at attention worker

NET_T_EXPERT_TO_ATTN  = 1.0  # fixed network delay expert -> attention
NET_T_ATTN_TO_EXPERT  = 1.0  # fixed network delay attention -> expert

RNG_SEED = 42

# Path to expert compute profile CSV (batch_size,avg_time_ms or generic time units)
# Default points to the included profile next to this file.
EXPERT_COMPUTE_PROFILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "expert_costs_profiles",
    "Qwen3-30B.csv",
)


# ------------------------------
# Token representation
# ------------------------------

@dataclass
class Token:
    tid: int
    birth_time: float
    # You can add more metadata here (e.g., layer history, expert id, etc.)


# ------------------------------
# Expert worker
# ------------------------------

class ExpertWorker:
    """
    Represents one expert worker in one layer.
    It owns `num_queues_per_worker` queues (for experts assigned to this worker).
    It always picks the *longest* queue to form the largest batch, up to MAX_BATCH_SIZE.
    Batch compute time depends on the formed batch size, using a loaded profile.
    Then each token in the batch is sent to the NEXT layer's attention worker
    (or marked finished if this is the last layer).
    """
    def __init__(self, env, layer_idx, worker_idx, num_queues_per_worker,
                 max_batch_size, compute_time_lookup,
                 net_t_expert_to_attn,
                 num_layers,
                 attention_layers,
                 completion_callback):
        self.env = env
        self.layer_idx = layer_idx
        self.worker_idx = worker_idx
        self.num_queues_per_worker = num_queues_per_worker
        self.max_batch_size = max_batch_size
        self._compute_time_lookup = compute_time_lookup
        self.net_t_expert_to_attn = net_t_expert_to_attn
        self.num_layers = num_layers
        self.attention_layers = attention_layers
        self.completion_callback = completion_callback

        # Per-expert queues
        self.queues = [deque() for _ in range(num_queues_per_worker)]

        # Event used for waking the worker when new work arrives
        self.has_work = env.event()

        # Start main process
        self.proc = env.process(self.run())

    @property
    def total_queue_length(self):
        return sum(len(q) for q in self.queues)

    def enqueue(self, local_queue_idx, token: Token):
        q = self.queues[local_queue_idx]
        was_empty = (self.total_queue_length == 0)
        q.append(token)
        # Wake the worker if it was idle
        if was_empty and not self.has_work.triggered:
            self.has_work.succeed()

    def run(self):
        while True:
            if self.total_queue_length == 0:
                # No work: go to sleep
                self.has_work = self.env.event()
                yield self.has_work

            # Pick the longest queue
            max_idx, max_q = max(
                enumerate(self.queues),
                key=lambda kv: len(kv[1])
            )
            if len(max_q) == 0:
                # Shouldn't really happen, but be defensive
                continue

            # Form a batch
            batch = []
            while max_q and len(batch) < self.max_batch_size:
                batch.append(max_q.popleft())

            # Expert compute
            compute_t = self._compute_time_lookup[len(batch)]
            yield self.env.timeout(compute_t)

            # Network delay to attention / completion
            yield self.env.timeout(self.net_t_expert_to_attn)

            # Route tokens onward
            for token in batch:
                if self.layer_idx == self.num_layers - 1:
                    # Last layer expert => token finishes here
                    self.completion_callback(token, self.env.now)
                else:
                    # Send to next layer's attention worker
                    next_attn = self.attention_layers[self.layer_idx + 1]
                    next_attn.enqueue(token)


# ------------------------------
# Attention + gating worker
# ------------------------------

class AttentionWorker:
    """
    Maintains a single FIFO queue.
    Each token:
      - waits in this queue
      - gets 1 tick "attention/gating" service
      - then after NET_T_ATTN_TO_EXPERT delay, is routed to an expert in THIS layer
        according to a uniform distribution over all experts.
    For layer 0, the source will inject tokens directly here.
    """
    def __init__(self, env, layer_idx,
                 total_expert_count,
                 ep_group_size,
                 attn_service_t,
                 net_t_attn_to_expert,
                 expert_layers):
        self.env = env
        self.layer_idx = layer_idx
        self.total_expert_count = total_expert_count
        self.ep_group_size = ep_group_size
        self.attn_service_t = attn_service_t
        self.net_t_attn_to_expert = net_t_attn_to_expert
        self.expert_layers = expert_layers

        self.queue = deque()
        self.has_work = env.event()

        # Derived:
        assert total_expert_count % ep_group_size == 0, \
            "total_expert_count must be divisible by ep_group_size"
        self.queues_per_worker = total_expert_count // ep_group_size

        self.proc = env.process(self.run())

    def enqueue(self, token: Token):
        was_empty = (len(self.queue) == 0)
        self.queue.append(token)
        if was_empty and not self.has_work.triggered:
            self.has_work.succeed()

    def run(self):
        while True:
            if len(self.queue) == 0:
                self.has_work = self.env.event()
                yield self.has_work

            token = self.queue.popleft()

            # Attention/gating compute
            yield self.env.timeout(self.attn_service_t)

            # Decide routing: pick an expert index uniformly
            global_expert_id = random.randint(0, self.total_expert_count - 1)
            worker_idx = global_expert_id // self.queues_per_worker
            local_queue_idx = global_expert_id % self.queues_per_worker

            # Network delay to expert queue
            yield self.env.timeout(self.net_t_attn_to_expert)

            # Send token into the chosen expert queue in THIS layer
            expert_worker = self.expert_layers[self.layer_idx][worker_idx]
            expert_worker.enqueue(local_queue_idx, token)


# ------------------------------
# Token source
# ------------------------------

def token_source(env, total_tokens, arrival_rate, first_attention: AttentionWorker):
    """
    Poisson arrival process with rate `arrival_rate`.
    Creates TOTAL_TOKENS tokens and injects them into the FIRST layer's
    attention worker queue.
    """
    for tid in range(total_tokens):
        t = Token(tid=tid, birth_time=env.now)
        first_attention.enqueue(t)

        # Poisson arrivals => Exp(lambda) inter-arrival
        inter_arrival = random.expovariate(arrival_rate)
        yield env.timeout(inter_arrival)


# ------------------------------
# Main simulation driver
# ------------------------------

def run_simulation(
    ep_group_size=EP_GROUP_SIZE,
    total_expert_count=TOTAL_EXPERT_COUNT,
    max_batch_size=MAX_BATCH_SIZE,
    num_layers=NUM_LAYERS,
    total_tokens=TOTAL_TOKENS,
    arrival_rate=ARRIVAL_RATE,
    expert_profile_path=EXPERT_COMPUTE_PROFILE_PATH,
):
    random.seed(RNG_SEED)
    env = simpy.Environment()

    # For metrics
    completion_times = {}  # tid -> completion_time

    def on_token_complete(token: Token, t_complete: float):
        completion_times[token.tid] = t_complete

    # Build layers: expert_layers[layer][worker]
    expert_layers = []
    attention_layers = []

    queues_per_worker = total_expert_count // ep_group_size

    # Load compute profile into a lookup table and validate coverage
    compute_time_lookup = expert_compute_time_lookup_table_from_profile(
        expert_profile_path, max_batch_size
    )

    # Build experts first with attention pointer set to None.
    # Then create attention workers and patch each expert's attention_layers reference.

    # Create expert workers
    for layer in range(num_layers):
        workers = []
        for w in range(ep_group_size):
            worker = ExpertWorker(
                env=env,
                layer_idx=layer,
                worker_idx=w,
                num_queues_per_worker=queues_per_worker,
                max_batch_size=max_batch_size,
                compute_time_lookup=compute_time_lookup,
                net_t_expert_to_attn=NET_T_EXPERT_TO_ATTN,
                num_layers=num_layers,
                attention_layers=None,  # temp, will fix after we create them
                completion_callback=on_token_complete,
            )
            workers.append(worker)
        expert_layers.append(workers)

    # Now create attention workers (they need expert_layers)
    for layer in range(num_layers):
        attn = AttentionWorker(
            env=env,
            layer_idx=layer,
            total_expert_count=total_expert_count,
            ep_group_size=ep_group_size,
            attn_service_t=ATTN_SERVICE_T,
            net_t_attn_to_expert=NET_T_ATTN_TO_EXPERT,
            expert_layers=expert_layers,
        )
        attention_layers.append(attn)

    # Now that we have attention_layers, fix the forward pointers in experts
    for layer in range(num_layers):
        for w in expert_layers[layer]:
            w.attention_layers = attention_layers

    # Create the source
    env.process(token_source(
        env,
        total_tokens=total_tokens,
        arrival_rate=arrival_rate,
        first_attention=attention_layers[0]
    ))

    # Run until all tokens are completed.
    # Easiest is to run until the event queue drains; since we only
    # generate TOTAL_TOKENS tokens and they all eventually complete,
    # the sim will naturally finish.
    env.run()

    # Metrics
    if len(completion_times) != total_tokens:
        print("WARNING: some tokens did not complete!", len(completion_times), "/", total_tokens)

    latencies = [
        completion_times[tid] - 0.0  # birth time is always 0.0 in this simple model,
                                     # since we don't track per-token birth separately here.
        for tid in completion_times
    ]
    avg_latency = sum(latencies) / len(latencies)
    makespan = max(completion_times.values())

    print(f"Simulation finished at time {makespan:.3f}")
    print(f"Average completion time over {len(latencies)} tokens: {avg_latency:.3f}")

    return {
        "completion_times": completion_times,
        "avg_latency": avg_latency,
        "makespan": makespan,
    }


if __name__ == "__main__":
    run_simulation()
