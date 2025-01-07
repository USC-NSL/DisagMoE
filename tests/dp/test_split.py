import torch
import random

bs = 256
hs = 4096

a = torch.zeros((bs, hs)).to("cuda:0", torch.bfloat16)

indices = list(range(bs))
random.shuffle(indices)
indices = torch.tensor(indices)

torch.cuda.synchronize()

st = torch.cuda.Event(enable_timing=True)
ed = torch.cuda.Event(enable_timing=True)

lst1 = torch.stack([a[i] for i in indices])

st.record()
lst1 = torch.stack([a[i] for i in indices])
ed.record()

torch.cuda.synchronize()
print(st.elapsed_time(ed))
indices = indices.to("cuda:0")

torch.cuda.synchronize()

st = torch.cuda.Event(enable_timing=True)
ed = torch.cuda.Event(enable_timing=True)

lst2 = torch.index_select(a, 0, indices)
torch.cuda.synchronize()

st.record()
lst2 = torch.index_select(a, 0, indices)
ed.record()

torch.cuda.synchronize()
print(st.elapsed_time(ed))

assert lst1.equal(lst2)