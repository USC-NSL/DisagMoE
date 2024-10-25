import torch
from disagmoe_c import *
import threading
from functools import partial
from disagmoe.utils.utils import get_nccl_unique_id as get_uid

uid = get_uid()

c1 = create_channel(0, 1, uid)
c2 = create_channel(1, 0, uid)

instantiate_channels([c1, c2])

t1 = torch.Tensor([1, 1, 1, 1]).type(dtype=torch.float16).cuda(device=0)
t2 = torch.Tensor([2, 2, 2, 2]).type(dtype=torch.float16).cuda(device=1)

meta = Metadata([4])

if False:
    # NOTE(hogura|20240927): this is not working since thread1 will block thread2 due to the stupid GIL
    thread1 = threading.Thread(target=partial(c1.send, t1.data_ptr(), meta))
    thread2 = threading.Thread(target=partial(c2.recv, t2.data_ptr(), meta))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

test_nccl_p2p(c1, t1.data_ptr(), c2, t2.data_ptr(), meta)

print(t1)
print(t2)