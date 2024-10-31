import threading
import time
import torch
import multiprocess

from zmq import Context, PUSH, PULL

st = 0
ed = 0

def sender():
    global st
    context = Context()
    socket = context.socket(PUSH)
    socket.bind("tcp://127.0.0.1:5555")
    data = torch.randn((128, 4096), dtype=torch.float).cpu().numpy().tobytes()
    st = time.time()
    socket.send(data)
    print(st)
    return st
    
def recver():
    global ed
    context = Context()
    socket = context.socket(PULL)
    socket.connect("tcp://127.0.0.1:5555")
    result = socket.recv()
    ed = time.time()
    print(ed)
    return ed
    
def test_thread():
    global st, ed
    t1 = threading.Thread(target=sender)
    t2 = threading.Thread(target=recver)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(ed - st)

def test_process():
    p1 = multiprocess.Process(target=sender)
    p2 = multiprocess.Process(target=recver)
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    print(ed - st)
    
test_process()