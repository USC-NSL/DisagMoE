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
        self.queuing_delays: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    def should_sample(self) -> bool:
        if not self.enabled:
            return False
        return random.random() < self.sample_rate

    def log_moe_step(self, batch_size: int, execution_time_ms: float):
        if not self.enabled:
            return
        self.moe_steps.append((batch_size, execution_time_ms, time.monotonic()))

    def log_queuing_delay(self, layer_id: int, expert_id: int, delay_ms: float):
        if not self.enabled:
            return
        self.queuing_delays[(layer_id, expert_id)].append(delay_ms)

    def get_data(self) -> Optional[dict]:
        """Return all collected data as a serializable dict (for cross-node collection)."""
        if not self.enabled:
            return None

        queuing_data = {}
        for (layer_id, expert_id), delays in self.queuing_delays.items():
            key = f"{layer_id}_{expert_id}"
            queuing_data[key] = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "delays_ms": delays,
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
        for (layer_id, expert_id), delays in self.queuing_delays.items():
            key = f"{layer_id}_{expert_id}"
            queuing_data[key] = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "delays_ms": delays,
                "mean_ms": sum(delays) / len(delays) if delays else 0,
                "count": len(delays),
            }
        with open(queuing_path, "w") as f:
            json.dump(queuing_data, f)

        return out_dir

    def reset(self):
        if not self.enabled:
            return
        self.moe_steps.clear()
        self.queuing_delays.clear()
