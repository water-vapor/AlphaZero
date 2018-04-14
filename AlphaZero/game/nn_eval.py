import atexit
import importlib
import traceback as tb
from queue import Empty as EmptyExc

import numpy as np
import yaml
import tensorflow as tf

# import AlphaZero.processing.go.state_converter as preproc
import AlphaZero.network.main as network
# import AlphaZero.env.go as go
from AlphaZero.train.parallel.util import *

with open('AlphaZero/config/game.yaml') as f:
    game_selection = yaml.load(f)['game']
with open('AlphaZero/config/' + game_selection + '.yaml') as c:
    game_config = yaml.load(c)
_preproc = importlib.import_module(game_config['state_converter_path'])
_state_tensor_converter = _preproc.StateTensorConverter(game_config)
_tensor_action_converter = _preproc.TensorActionConverter(game_config)


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

    def __init__(self, cluster, game_config, ext_config):  # TODO: use proper default value
        """
        :param net: network class. This class doesn't create network.
        :param max_batch_size: Int
        """
        printlog('create nn_eval')
        self.cluster = cluster
        self.job = ext_config['job']
        self.max_batch_size = ext_config['max_batch_size']
        self.load_path = ext_config.get('load_path')
        self.num_gpu = ext_config['num_gpu']
        atexit.register(kill_children)  # kill all the children when the program exit
        self.listen_proc = mp.Process(target=self.listen, name=self.job + '_nn_eval')

        self.rwlock = RWLock()

        self.rcpt = Reception(self.max_batch_size * 2)
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
        state_np = _state_tensor_converter.state_to_tensor(state)
        result_np = self.rcpt.req(state_np)
        # This game specific conversation is implemented in state converter
        result = (_tensor_action_converter.tensor_to_action(result_np[0]), result_np[1])
        # for i in range(361):
        #     result[0].append(((i // 19, i % 19), result_np[0][i]))
        # result[0].append((go.PASS_MOVE, result_np[0][361]))
        return result

    def sl_listen(self):
        printlog_thrd('start')
        while True:
            (req_type, filename), s_conn = self.sl_rcpt.get()
            if req_type == 'load':
                printlog_thrd('load')
                self.rwlock.w_acquire()
                self.net.load(filename)
                self.rwlock.w_release()
                printlog_thrd('load complete')
                s_conn.send('done')
            elif req_type == 'save':
                printlog_thrd('save')
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
        self.net = network.Network(game_config, num_gpu=self.num_gpu,
                                   cluster=self.cluster, job=self.job)
        if self.load_path is not None:
            printlog('load model')
            self.net.load(self.load_path)

        thrd.Thread(target=self.sl_listen, name='sl_listener').start()

        printlog('loop begin')
        while True:
            try:
                reqs = []
                for i in range(self.max_batch_size):
                    block = i < self.num_gpu
                    reqs.append(self.rcpt.get(block))
            except EmptyExc:
                pass
            finally:
                # printlog(len(reqs), 'reqs')
                states_np = np.concatenate([req[0] for req in reqs], 0)
                rp, rv = self.net.response((states_np,))
                for i in range(len(reqs)):
                    reqs[i][1].send((rp[i], rv[i]))
