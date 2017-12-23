import multiprocessing as mp
import threading as thrd

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

    def __init__(self, cluster, job, load_file=None, max_batch_size=32, **kwargs):  # TODO: use proper default value
        """
        :param net: Network class. This class doesn't create Network.
        :param max_batch_size: Int
        """
        printlog('create nn_eval')
        self.cluster = cluster
        self.job = job
        self.max_batch_size = max_batch_size
        self.load_file = load_file
        self.net = None
        atexit.register(kill_children)  # kill all the children when the program exit
        self.listen_proc = mp.Process(target=self.listen, name=kwargs.get('name', 'nn_eval')+'_listener')

        self.rwlock = RWLock()

        self.rcpt = Reception(max_batch_size*2)
        self.sl_rcpt = Reception(5)

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
        result_np = self.rcpt.req(state_np)
        result = ([], result_np[1])
        for i in range(361):
            result[0].append(( (i//19, i%19), result_np[0][i] ))
        result[0].append(( go.PASS_MOVE, result_np[0][361] ))
        return result

    def sl_listen(self):
        printlog_thrd('start')
        while True:
            (req_type, filename), s_conn = self.sl_rcpt.get()
            if req_type == 'load':
                self.rwlock.w_acquire()
                self.net.load(filename)
                self.rwlock.w_release()
                s_conn.send('done')
            elif req_type == 'save':
                self.net.save(filename)
                s_conn.send('done')

    def load(self, filename):
        self.sl_rcpt.req(('load', filename))

    def save(self, filename):
        self.sl_rcpt.req(('save', filename))

    def listen(self):
        """
        This function is run in another thread launched by main evaluator/generator thread. There will be only 2
        listeners. They are NN to be evaluated and best NN so far.
        """
        printlog('create network')
        self.net = network.Network(config_file="AlphaGoZero/Network/reinforce.yaml", cluster=self.cluster, job=self.job)
        if self.load_file is not None:
            self.net.load(self.load_file)

        thrd.Thread(target=self.sl_listen, name='sl_listener').start()

        printlog('loop begin')
        while True:
            try:
                reqs = []
                for i in range(self.max_batch_size):
                    block = i==0
                    reqs.append(self.rcpt.get(block))
            except EmptyExc:
                pass
            finally:
                # printlog(len(reqs), 'reqs')
                states_np = np.concatenate([req[0] for req in reqs], 0)
                rp, rv = self.net.response((states_np,))
                for i in range(len(reqs)):
                    reqs[i][1].send((rp[i], rv[i]))
