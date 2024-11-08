import ray

@ray.remote
class Worker:
    
    def __init__(self):
        self.val = 1
        
    @property
    def a(self):
        return self.val
    

w = Worker.remote()

print(ray.get(w.a.remote()))