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
    
    def __init__(self, rate: float, cv: float, fixed_input_len = -1):
        self.rate = rate
        self.cv = cv
        self.fixed_input_len = fixed_input_len
        
    def generate_arrivals(self, n_request: int) -> List[int]:
        raise NotImplementedError()
    
    def generate_input_lens(self, n_request: int) -> List[int]:
        raise NotImplementedError()
    
    def generate(self, n_request: int) -> Workload:
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
    

def get_generator(name) -> Generator:
    return {
        "uniform": UniformGenerator,
        "poisson": PoissonGenerator
    }[name]
    

if __name__ == "__main__":
    generator = PoissonGenerator(1, 1, 128)
    workload = generator.generate(10)
    print(workload.arrivals)