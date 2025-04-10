import numpy as np
from typing import List, override


class Workload:
    
    def __init__(self, arrivals, input_lens = [], fixed_input_len = -1):
        self.arrivals = arrivals
        self.input_lens = input_lens
        self.fixed_input_len = fixed_input_len
        
    def __getitem__(self, index):
        if self.fixed_input_len == -1:
            return self.arrivals[index], self.input_lens[index]
        else:
            return self.arrivals[index], self.fixed_input_len
    
    def __iter__(self):
        for i in range(len(self.arrivals)):
            yield self[i]

class Generator:
    
    def __init__(self, rate: int, cv: float, fixed_input_len = -1):
        self.rate = rate
        self.cv = cv
        self.fixed_input_len = fixed_input_len
        
    def generate_arrivals(self, n_request: int) -> List[int]:
        raise NotImplementedError()
    
    def generate_input_lens(self, n_request: int) -> List[int]:
        raise NotImplementedError()
    
    def get_num_requests(self, duration: int) -> int:
        return duration * self.rate
        
    def generate(self, duration: int) -> Workload:
        n_request = self.get_num_requests(duration)
        arrivals = self.generate_arrivals(n_request)
        input_lens = self.generate_input_lens(n_request) if self.fixed_input_len == -1 else []
        print("Using Workload Generator:", self.__class__.__name__, 
              f"generated {n_request} requests, maximal arrival {arrivals[-1]}s.")
        return Workload(arrivals, input_lens, self.fixed_input_len)


class UniformGenerator(Generator):
    
    @override
    def generate_arrivals(self, n_request: int) -> List[int]:
        gap = 1 / self.rate
        arrivals = [gap * i for i in range(n_request)]
        return arrivals
    
    
class PoissonGenerator(Generator):
    
    @override
    def generate_arrivals(self, n_request: int) -> List[int]:
        gap = np.random.exponential(1 / self.rate, n_request)
        arrivals = np.cumsum(gap)
        return arrivals
    
class OfflineGenerator(Generator):
    
    @override
    def generate_arrivals(self, n_request: int) -> List[int]:
        arrivals = np.zeros(n_request)
        return arrivals
    
class IncrementalPoissonGenerator(Generator):
    
    increment: int = 2 # rate increment
    interval: int = 60 # by seconds
    
    @override
    def get_num_requests(self, duration):
        num_reqs = 0
        rate = self.rate
        while duration > self.interval:
            elapse = min(self.interval, duration)
            num_reqs += int(elapse * rate)
            duration -= elapse
            rate += self.increment
        return num_reqs
    
    @override
    def generate_arrivals(self, n_request: int) -> List[float]:
        
        arrivals = []
        current_time = 0.0
        total_reqs = 0
        rate = self.rate
        while total_reqs < n_request:
            num_reqs = min(rate * self.interval, n_request - total_reqs)
            gap = np.random.exponential(1 / self.rate, int(num_reqs))
            step_arrivals = np.cumsum(gap)
            arrivals = np.concatenate((arrivals, step_arrivals + current_time))
            rate += self.increment
            current_time += self.interval
            total_reqs += num_reqs
        
        return arrivals
    

def get_generator(name) -> Generator:
    return {
        "uniform": UniformGenerator,
        "poisson": PoissonGenerator,
        "offline": OfflineGenerator,
        "incremental_poisson": IncrementalPoissonGenerator,
    }[name]
    

if __name__ == "__main__":
    generator = PoissonGenerator(1, 1, 128)
    workload = generator.generate(10)
    print(workload.arrivals)