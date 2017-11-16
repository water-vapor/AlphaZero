import multiprocessing as mp
import atexit
from queue import Empty as EmptyExc
import traceback as tb
import AlphaGoZero.preprocessing as preproc
import numpy as np

class _EvalReq:
    """
    This class encapsulates game state to be evaluated, result and semaphore for blocking receive
    """

    def __init__(self, state):
        self.state = state
        self.sem = mp.Semaphore(0)
        self.result = None

def kill_children():
    for p in mp.active_children():
        p.terminate()

class NNEvaluator:
    """
    Provide neural network evaluation services for model evaluator and data generator. Instances should be created by
    the main evaluator/generator thread. Context manager (with statement) is preferred because of the automatic start
    and termination of the listening thread.
    e.g.
        with NNEvaluator(...) as eval:
            pass
    """

    def __init__(self, net, max_batch_size):  # TODO: add default value
        """
        :param net: Network class. This class doesn't create Network.
        :param max_batch_size: Int
        """
        self.max_batch_size = max_batch_size
        self.queue = mp.Queue(max_batch_size)
        self.net = net
        atexit.register(kill_children)  # kill all the children when the program exit
        self.listen_proc = mp.Process(target=self.listen)

    def __enter__(self):
        """Will be called where the "with" statement begin"""
        self.listen_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Will be called where the "with" statement end"""
        self.listen_proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def eval(self, state):
        """
        This function is called by mcts threads. There will be multiple function calls
        :param state: GameState
        :return: (policy, value) pair
        """
        state_np = preproc.Preprocess().state_to_tensor(state)
        req = _EvalReq(state_np)
        self.queue.put(req)
        req.sem.acquire()
        return req.result

    def listen(self):
        """
        This function is run in another thread launched by main evaluator/generator thread. There will be only 2
        listeners for NN to be evaluated and best NN so far, respectively.
        """
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
