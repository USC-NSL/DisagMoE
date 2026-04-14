import json
import os
import random
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class AdvancedLogger:
    def __init__(self, enabled: bool, output_dir: str, device_id: int, sample_rate: float = 0.1):
        self.enabled = enabled
        self.output_dir = output_dir
        self.device_id = device_id
        self.sample_rate = sample_rate

        if not enabled:
            return

        # Each entry: (batch_size, execution_time_ms, timestamp_s)
        self.moe_steps: List[Tuple[int, float, float]] = []
        # Each entry: (delay_ms, timestamp_s)
        self.queuing_delays: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
        # Each entry: (timestamp_s, scheduled_layer_id, layer_depths: List[int])
        self.queue_snapshots: List[Tuple[float, int, List[int]]] = []

        # Dispatcher put operations: (num_tokens, latency_ms, layer_id, timestamp_s)
        self.dispatcher_puts: List[Tuple[int, float, int, float]] = []
        # Pool put operations (new request admission): (num_tokens, timestamp_s)
        self.pool_puts: List[Tuple[int, float]] = []
        # Receive completions: (peer_id, layer_id, num_tokens, num_bytes, posted_ts_s, completed_ts_s, is_local)
        self.recv_completions: List[Tuple[int, int, int, int, float, float, bool]] = []
        # Pending-send stalls: (start_ts_s, end_ts_s, pending_before, max_pending, yield_count)
        self.pending_send_stalls: List[Tuple[float, float, int, int, int]] = []

    def should_sample(self) -> bool:
        if not self.enabled:
            return False
        return random.random() < self.sample_rate

    def log_moe_step(self, batch_size: int, execution_time_ms: float):
        if not self.enabled:
            return
        self.moe_steps.append((batch_size, execution_time_ms, time.time()))

    def log_queuing_delay(self, layer_id: int, expert_id: int, delay_ms: float, timestamp_s: float = 0.0):
        if not self.enabled:
            return
        self.queuing_delays[(layer_id, expert_id)].append((delay_ms, timestamp_s))

    def log_queue_snapshot(self, timestamp_s: float, scheduled_layer_id: int, layer_depths: List[int]):
        if not self.enabled:
            return
        self.queue_snapshots.append((timestamp_s, scheduled_layer_id, layer_depths))

    def log_dispatcher_put(self, num_tokens: int, latency_ms: float, layer_id: int, timestamp_s: float = 0.0):
        if not self.enabled:
            return
        self.dispatcher_puts.append((num_tokens, latency_ms, layer_id, timestamp_s or time.time()))

    def log_pool_put(self, num_tokens: int, timestamp_s: float = 0.0):
        if not self.enabled:
            return
        self.pool_puts.append((num_tokens, timestamp_s or time.time()))

    def log_recv_completion(
        self,
        peer_id: int,
        layer_id: int,
        num_tokens: int,
        num_bytes: int,
        posted_ts_s: float,
        completed_ts_s: float,
        is_local: bool,
    ):
        if not self.enabled:
            return
        self.recv_completions.append(
            (peer_id, layer_id, num_tokens, num_bytes, posted_ts_s, completed_ts_s, is_local)
        )

    def log_pending_send_stall(
        self,
        start_ts_s: float,
        end_ts_s: float,
        pending_before: int,
        max_pending: int,
        yield_count: int,
    ):
        if not self.enabled:
            return
        self.pending_send_stalls.append(
            (start_ts_s, end_ts_s, pending_before, max_pending, yield_count)
        )

    def get_data(self) -> Optional[dict]:
        """Return all collected data as a serializable dict (for cross-node collection)."""
        if not self.enabled:
            return None

        queuing_data = {}
        for (layer_id, expert_id), entries in self.queuing_delays.items():
            key = f"{layer_id}_{expert_id}"
            delays = [e[0] for e in entries]
            timestamps = [e[1] for e in entries]
            queuing_data[key] = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "delays_ms": delays,
                "timestamps_s": timestamps,
                "mean_ms": sum(delays) / len(delays) if delays else 0,
                "count": len(delays),
            }

        return {
            "device_id": self.device_id,
            "moe_steps": {
                "batch_sizes": [s[0] for s in self.moe_steps],
                "execution_times_ms": [s[1] for s in self.moe_steps],
                "timestamps_s": [s[2] for s in self.moe_steps],
            },
            "queuing_delays": queuing_data,
            "queue_snapshots": {
                "timestamps_s": [s[0] for s in self.queue_snapshots],
                "scheduled_layer_ids": [s[1] for s in self.queue_snapshots],
                "layer_depths": [s[2] for s in self.queue_snapshots],
            },
            "dispatcher_puts": {
                "num_tokens": [d[0] for d in self.dispatcher_puts],
                "latencies_ms": [d[1] for d in self.dispatcher_puts],
                "layer_ids": [d[2] for d in self.dispatcher_puts],
                "timestamps_s": [d[3] for d in self.dispatcher_puts],
            },
            "pool_puts": {
                "num_tokens": [p[0] for p in self.pool_puts],
                "timestamps_s": [p[1] for p in self.pool_puts],
            },
            "recv_completions": {
                "peer_ids": [r[0] for r in self.recv_completions],
                "layer_ids": [r[1] for r in self.recv_completions],
                "num_tokens": [r[2] for r in self.recv_completions],
                "num_bytes": [r[3] for r in self.recv_completions],
                "posted_timestamps_s": [r[4] for r in self.recv_completions],
                "completed_timestamps_s": [r[5] for r in self.recv_completions],
                "is_local": [r[6] for r in self.recv_completions],
            },
            "pending_send_stalls": {
                "start_timestamps_s": [s[0] for s in self.pending_send_stalls],
                "end_timestamps_s": [s[1] for s in self.pending_send_stalls],
                "pending_before": [s[2] for s in self.pending_send_stalls],
                "max_pending": [s[3] for s in self.pending_send_stalls],
                "yield_counts": [s[4] for s in self.pending_send_stalls],
            },
        }

    def dump(self, suffix: str = "") -> Optional[str]:
        if not self.enabled:
            return None

        out_dir = os.path.join(self.output_dir, f"device_{self.device_id}")
        os.makedirs(out_dir, exist_ok=True)

        moe_path = os.path.join(out_dir, f"moe_steps{suffix}.json")
        with open(moe_path, "w") as f:
            json.dump(
                {
                    "batch_sizes": [s[0] for s in self.moe_steps],
                    "execution_times_ms": [s[1] for s in self.moe_steps],
                    "timestamps_s": [s[2] for s in self.moe_steps],
                },
                f,
            )

        queuing_path = os.path.join(out_dir, f"queuing_delays{suffix}.json")
        queuing_data = {}
        for (layer_id, expert_id), entries in self.queuing_delays.items():
            key = f"{layer_id}_{expert_id}"
            delays = [e[0] for e in entries]
            timestamps = [e[1] for e in entries]
            queuing_data[key] = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "delays_ms": delays,
                "timestamps_s": timestamps,
                "mean_ms": sum(delays) / len(delays) if delays else 0,
                "count": len(delays),
            }
        with open(queuing_path, "w") as f:
            json.dump(queuing_data, f)

        queue_snapshot_path = os.path.join(out_dir, f"queue_snapshots{suffix}.json")
        with open(queue_snapshot_path, "w") as f:
            json.dump(
                {
                    "timestamps_s": [s[0] for s in self.queue_snapshots],
                    "scheduled_layer_ids": [s[1] for s in self.queue_snapshots],
                    "layer_depths": [s[2] for s in self.queue_snapshots],
                },
                f,
            )

        dispatcher_path = os.path.join(out_dir, f"dispatcher_puts{suffix}.json")
        with open(dispatcher_path, "w") as f:
            json.dump(
                {
                    "num_tokens": [d[0] for d in self.dispatcher_puts],
                    "latencies_ms": [d[1] for d in self.dispatcher_puts],
                    "layer_ids": [d[2] for d in self.dispatcher_puts],
                    "timestamps_s": [d[3] for d in self.dispatcher_puts],
                },
                f,
            )

        pool_puts_path = os.path.join(out_dir, f"pool_puts{suffix}.json")
        with open(pool_puts_path, "w") as f:
            json.dump(
                {
                    "num_tokens": [p[0] for p in self.pool_puts],
                    "timestamps_s": [p[1] for p in self.pool_puts],
                },
                f,
            )

        recv_completions_path = os.path.join(out_dir, f"recv_completions{suffix}.json")
        with open(recv_completions_path, "w") as f:
            json.dump(
                {
                    "peer_ids": [r[0] for r in self.recv_completions],
                    "layer_ids": [r[1] for r in self.recv_completions],
                    "num_tokens": [r[2] for r in self.recv_completions],
                    "num_bytes": [r[3] for r in self.recv_completions],
                    "posted_timestamps_s": [r[4] for r in self.recv_completions],
                    "completed_timestamps_s": [r[5] for r in self.recv_completions],
                    "is_local": [r[6] for r in self.recv_completions],
                },
                f,
            )

        pending_stalls_path = os.path.join(out_dir, f"pending_send_stalls{suffix}.json")
        with open(pending_stalls_path, "w") as f:
            json.dump(
                {
                    "start_timestamps_s": [s[0] for s in self.pending_send_stalls],
                    "end_timestamps_s": [s[1] for s in self.pending_send_stalls],
                    "pending_before": [s[2] for s in self.pending_send_stalls],
                    "max_pending": [s[3] for s in self.pending_send_stalls],
                    "yield_counts": [s[4] for s in self.pending_send_stalls],
                },
                f,
            )

        return out_dir

    def reset(self):
        if not self.enabled:
            return
        self.moe_steps.clear()
        self.queuing_delays.clear()
        self.queue_snapshots.clear()
        self.dispatcher_puts.clear()
        self.pool_puts.clear()
        self.recv_completions.clear()
        self.pending_send_stalls.clear()
