import multiprocessing as mp
import atexit
from queue import Empty as EmptyExc
import traceback as tb
import AlphaGoZero.preprocessing as preproc
import numpy as np

class _EvalReq:

    def __init__(self, state):
        self.state = state
        self.sem = mp.Semaphore(0)
        self.result = None

def kill_children():
    for p in mp.active_children():
        p.terminate()

class NNEvaluator:

    def __init__(self, net, max_batch_size):
        self.max_batch_size = max_batch_size
        self.queue = mp.Queue(max_batch_size)
        self.net = net
        atexit.register(kill_children)
        self.listen_proc = mp.Process(target=self.listen)

    def __enter__(self):
        self.listen_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.listen_proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def eval(self, state):
        state_np = preproc.Preprocess().state_to_tensor(state)
        req = _EvalReq(state_np)
        self.queue.put(req)
        req.sem.acquire()
        return req.result

    def listen(self):
        while True:
            try:
                reqs = []
                for i in range(self.max_batch_size):
                    block = i==0
                    reqs.append(self.queue.get(block))
            except EmptyExc:
                pass
            finally:
                reqs_np = np.stack(reqs, 0)
                rp, rv = self.net.response(reqs_np)
                for i in range(len(reqs)):
                    reqs[i].result = (rp[i], rv[i])
                    reqs[i].sem.release()
