from disagmoe_c import *

batch = TensorBatch()

dispatcher = MuAttnDispatcher([1], 1)
dispatcher.start()
dispatcher.put(batch)