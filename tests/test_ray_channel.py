import ray
from disagmoe.utils.utils import get_nccl_unique_id

@ray.remote(num_cpus=1, num_gpus=1)
class Worker:
    def __init__(self):
        self.channel = None
    
    def init(self, local, peer, uid):
        from disagmoe_c import create_channel_py_map
        print(uid)
        self.channel = create_channel_py_map(local, peer, {peer: uid})
        
    def inst(self):
        self.channel.instantiate()
        

uid = get_nccl_unique_id()
print(uid)

A = Worker.remote()
B = Worker.remote()

ray.get([A.init.remote(0, 1, uid), B.init.remote(1, 0, uid)])

ray.get([
    A.inst.remote(),
    B.inst.remote()
])

print("passed")