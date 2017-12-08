import multiprocessing as mp
import atexit
from queue import Empty as EmptyExc
import traceback as tb
import AlphaGoZero.preprocessing.preprocessing as preproc
import AlphaGoZero.Network.main as network
import AlphaGoZero.go as go
from AlphaGoZero.reinforcement_learning.parallel.util import *
import numpy as np

class _EvalReq:
    """
    This class encapsulates game state to be evaluated and pipe
    """

    def __init__(self, state_np, conn):
        self.state_np = state_np
        self.conn = conn

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

    def __init__(self, load_file=None, max_batch_size=32, **kwargs):  # TODO: use proper default value
        """
        :param net: Network class. This class doesn't create Network.
        :param max_batch_size: Int
        """
        printlog('create nn_eval')
        self.max_batch_size = max_batch_size
        self.queue = mp.Queue(max_batch_size)
        self.load_file = load_file
        self.net = None
        atexit.register(kill_children)  # kill all the children when the program exit
        self.listen_proc = mp.Process(target=self.listen, name=kwargs.get('name', 'nn_eval')+'_listener')
        self.active_game = mp.Semaphore(0)
        self.loading = mp.Lock()

        self.conn_queue = mp.Queue(max_batch_size*2)
        for i in range(max_batch_size*2):
            self.conn_queue.put(mp.Pipe())

    def __enter__(self):
        """Will be called where the "with" statement begin"""
        printlog('nn_eval: start listening')
        self.listen_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Will be called where the "with" statement end"""
        printlog('nn_eval: terminate listener')
        self.listen_proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def eval(self, state):
        """
        This function is called by mcts threads. There will be multiple function calls
        :param state: GameState
        :return: (policy, value) pair
        """
        state_np = preproc.Preprocess().state_to_tensor(state) # TODO: check preprocessor
        r_conn, s_conn = self.conn_queue.get()
        req = _EvalReq(state_np, s_conn)
        self.queue.put(req)
        result_np = r_conn.recv()
        self.conn_queue.put((r_conn, s_conn))
        result = ([], result_np[1])
        for i in range(361):
            result[0].append(( (i//19, i%19), result_np[0][i] ))
        result[0].append(( go.PASS_MOVE, result_np[0][361] ))
        return result

    def load(self, filename):
        self.loading.acquire()      # loading lock

        self.active_game.acquire()  # check whether there are active games
        self.active_game.release()  # release semaphore

        self.net.load(filename)

        self.loading.release()      # release lock

    def save(self, filename):
        self.net.save(filename)

    def listen(self):
        """
        This function is run in another thread launched by main evaluator/generator thread. There will be only 2
        listeners. They are NN to be evaluated and best NN so far.
        """
        printlog('create network')
        self.net = network.Network(config_file="AlphaGoZero/Network/reinforce.yaml")
        if self.load_file is not None:
            self.net.load(self.load_file)
        printlog('loop begin')
        while True:
            try:
                reqs = []
                for i in range(self.max_batch_size):
                    block = i==0
                    reqs.append(self.queue.get(block))
            except EmptyExc:
                pass
            finally:
                printlog(len(reqs), 'reqs')
                states_np = np.concatenate([req.state_np for req in reqs], 0)
                rp, rv = self.net.response((states_np,))
                for i in range(len(reqs)):
                    reqs[i].conn.send((rp[i], rv[i]))
