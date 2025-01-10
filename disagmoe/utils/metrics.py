from dataclasses import dataclass

@dataclass
class Metric:
    effective_tokens: float = 0
    queueing_tokens: float = 0
    queueing_batches: float = 0
    t_schedule: float = 0
    t_step: float = 0
    t_preprocess: float = 0
    t_execute: float = 0
    t_postprocess: float = 0
    
    counter: int = 0
    
    def step(self):
        self.counter += 1
    
    def update(self, filed_name: str, value):
        value_old = getattr(self, filed_name)
        setattr(self, filed_name, (value_old * (self.counter - 1) + value) / self.counter)