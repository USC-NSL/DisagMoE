import numpy as np
from typing import List, override


class Workload:
    
    def __init__(self, arrivals, input_lens, output_lens):
        self.arrivals = arrivals
        self.input_lens = input_lens
        self.output_lens = output_lens
        
    def __getitem__(self, index):
        return self.arrivals[index], self.input_lens[index], self.output_lens[index]
    
    def __iter__(self):
        for i in range(len(self.arrivals)):
            yield self[i]

class Generator:
    
    def __init__(self, rate: int, cv: float, min_input_len, max_input_len, min_output_len, max_output_len):
        self.rate = rate
        self.cv = cv
        self.min_input_len = min_input_len
        self.max_input_len = max_input_len
        self.min_output_len = min_output_len
        self.max_output_len = max_output_len
        
    def generate_arrivals(self, n_request: int) -> List[int]:
        raise NotImplementedError()
    
    def generate_input_lens(self, n_request: int) -> List[int]:
        return np.random.randint(self.min_input_len, self.max_input_len, n_request)
    
    def generate_output_lens(self, n_request: int) -> List[int]:
        return np.random.randint(self.min_output_len, self.max_output_len, n_request)
    
    def get_num_requests(self, duration: int) -> int:
        return duration * self.rate
        
    def generate_duration(self, duration: int) -> Workload:
        n_request = self.get_num_requests(duration)
        return self.generate_num(n_request)
    
    def generate_num(self, num: int) -> Workload:
        arrivals = self.generate_arrivals(num)
        input_lens = self.generate_input_lens(num)
        output_lens = self.generate_output_lens(num)
        print("Using Workload Generator:", self.__class__.__name__, 
              f"generated {num} requests, maximal arrival {arrivals[-1]}s.")
        return Workload(arrivals, input_lens, output_lens)

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
    
    increment: int = 200 # rate increment
    interval: int = 60 # by seconds
    
    @override
    def get_num_requests(self, duration):
        num_reqs = 0
        rate = self.rate
        while duration > 0:
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
            gap = np.random.exponential(1 / rate, int(num_reqs))
            step_arrivals = np.cumsum(gap) + current_time
            arrivals = np.concatenate((arrivals, step_arrivals))
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