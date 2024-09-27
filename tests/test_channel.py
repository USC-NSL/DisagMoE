from disagmoe_c import *

uid = get_nccl_unique_id()

c1 = create_channel(0, 1, uid)
c2 = create_channel(1, 0, uid)

instantiate_channels([c1, c2])