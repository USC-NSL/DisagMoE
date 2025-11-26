import torch
import time
import os
import threading
from disagmoe.utils.logger import get_logger

class EngineProfilerMixin:
    
    profile_start_min_batch_size: int = 0
    profile_num_steps: int = 0
    profile_dir: str = None
    profile_in_progress: bool = False
    profile_step_count: int = 0
    lock: threading.Lock = threading.Lock()
    
    def init_profile(self, profile_start_min_batch_size: int, profile_num_steps: int, profile_dir=None):
        with self.lock:
            self.profile_start_min_batch_size = profile_start_min_batch_size
            self.profile_num_steps = profile_num_steps
            self.profile_in_progress = False
            self.profile_step_count = 0
            
            if profile_dir is None:
                get_logger().info("profiling directory not specified, using default")
                profile_dir = os.environ.get("DMOE_PROFILE_DIR", "torch_profile")
                
            try:
                os.makedirs(profile_dir, exist_ok=True)
            except Exception as e:
                get_logger().warning(f"failed to create profile dir '{profile_dir}': {e}, falling back to 'torch_profile'")
                profile_dir = "torch_profile"
                os.makedirs(profile_dir, exist_ok=True)
            
            get_logger().info(f"Enable profiler, results stored at {profile_dir}")
            self.profile_dir = profile_dir
            
    def step_profile(self, batch_size: int):
        with self.lock:
            if not self.profile_in_progress:
                if self.profile_num_steps > 0 and batch_size >= self.profile_start_min_batch_size:
                    self.start_profile()
            else:
                self.profile_step_count += 1
                if self.profile_step_count >= self.profile_num_steps:
                    self.stop_profile()
        
    def start_profile(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        self.profiler.start()
        self.profile_in_progress = True

    def stop_profile(self):
        if self.profiler is None:
            return
        try:
            self.profiler.stop()
            ts = int(time.time())
            out_file = os.path.join(self.profile_dir or ".", f"engine-{self.device_id}.{ts}.pt.trace.json")
            try:
                self.profiler.export_chrome_trace(out_file)
                get_logger().info(f"exported chrome trace to {out_file}")
            except Exception as ee:
                get_logger().warning(f"failed to export chrome trace: {ee}")
        except RuntimeError as e:
            # Profiler may already be stopped if the schedule ended; make stop idempotent.
            get_logger().warning(f"profiler.stop() ignored: {e}")
        finally:
            self.profiler = None
            self.profile_in_progress = False
            self.profile_num_steps = 0