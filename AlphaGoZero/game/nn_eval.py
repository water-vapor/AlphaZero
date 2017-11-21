import multiprocessing as mp
import atexit
from queue import Empty as EmptyExc
import traceback as tb
import AlphaGoZero.preprocessing.preprocessing as preproc
import AlphaGoZero.Network.main as network
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

    def __init__(self, load_file=None, max_batch_size=32):  # TODO: use proper default value
        """
        :param net: Network class. This class doesn't create Network.
        :param max_batch_size: Int
        """
        print('create nn_eval')
        self.max_batch_size = max_batch_size
        self.queue = mp.Queue(max_batch_size)
        self.load_file = load_file
        self.net = None
        atexit.register(kill_children)  # kill all the children when the program exit
        self.listen_proc = mp.Process(target=self.listen)
        self.active_game = mp.Semaphore(0)
        self.loading = mp.Lock()

        self.conn_queue = mp.Queue(max_batch_size)
        for i in range(max_batch_size):
            self.conn_queue.put(mp.Pipe())

    def __enter__(self):
        """Will be called where the "with" statement begin"""
        print('nn_eval: start listening')
        self.listen_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Will be called where the "with" statement end"""
        print('nn_eval: terminate listener')
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
        result = r_conn.recv()
        self.conn_queue.put((r_conn, s_conn))
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
        print('nn_eval: create network')
        self.net = network.Network(config_file="AlphaGoZero/Network/reinforce.yaml")
        if self.load_file is not None:
            self.net.load(self.load_file)
        print('nn_eval: loop begin')
        while True:
            try:
                reqs = []
                for i in range(self.max_batch_size):
                    block = i==0
                    reqs.append(self.queue.get(block))
            except EmptyExc:
                pass
            finally:
                states_np = np.concatenate([req.state_np for req in reqs], 0)
                rp, rv = self.net.response((states_np,))
                for i in range(len(reqs)):
                    reqs[i].conn.send((rp[i], rv[i]))
