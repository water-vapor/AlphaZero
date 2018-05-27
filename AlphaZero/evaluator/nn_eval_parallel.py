import os
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

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'game.yaml')) as f:
    game_selection = yaml.load(f)['game']
with open(os.path.join(os.path.dirname(__file__), '..', 'config', game_selection + '.yaml')) as c:
    game_config = yaml.load(c)
_preproc = importlib.import_module(game_config['state_converter_path'])
_state_tensor_converter = _preproc.StateTensorConverter(game_config)
_tensor_action_converter = _preproc.TensorActionConverter(game_config)


def kill_children():
    for p in mp.active_children():
        p.terminate()


class NNEvaluator:
    """
    Provide neural network evaluation services for model evaluator and data generator. Instances should be created by
    the main evaluator/generator thread. Context manager (with statement) is preferred because of the automatic start
    and termination of the listening thread.

    Example:

        with NNEvaluator(...) as eval:
            pass

    Args:
        cluster: Tensorflow cluster spec
        game_config: A dictionary of game environment configuration
        ext_config: A dictionary of system configuration
    """

    def __init__(self, cluster, game_config, ext_config):  # TODO: use proper default value

        printlog('create nn_eval')
        self.cluster = cluster
        self.job = ext_config['job']
        self.max_batch_size = ext_config['max_batch_size']
        self.load_path = ext_config.get('load_path')
        self.num_gpu = ext_config['num_gpu']
        atexit.register(kill_children)  # kill all the children when the program exit
        self.listener = mp.Process(target=self.listen, name=self.job + '_nn_eval')

        self.rwlock = RWLock()

        self.server_client_conn = ServerClientConn(self.max_batch_size * 2)
        self.save_load_conn = ServerClientConn(5)

    def __enter__(self):
        """Will be called where the "with" statement begin"""
        printlog('nn_eval: start listening')
        self.listener.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Will be called where the "with" statement end"""
        printlog('nn_eval: terminate listener')
        self.listener.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def eval(self, state):
        """
        This function is called by mcts threads.

        Args:
            state: GameState

        Returns:
            Tuple: (policy, value) pair
        """
        state_np = _state_tensor_converter.state_to_tensor(state)
        result_np = self.server_client_conn.req(state_np)
        # This game specific conversation is implemented in state converter
        result = (_tensor_action_converter.tensor_to_action(result_np[0]), result_np[1])
        # for i in range(361):
        #     result[0].append(((i // 19, i % 19), result_np[0][i]))
        # result[0].append((go.PASS_MOVE, result_np[0][361]))
        return result

    def sl_listen(self):
        """
        The listener for saving and loading the network parameters. This is run in new thread instead of process.
        """
        printlog_thrd('start')
        while True:
            (req_type, filename), s_conn = self.save_load_conn.get()
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
        """
        Send the load request.

        Args:
            filename: the filename of the checkpoint
        """
        self.save_load_conn.req(('load', filename))

    def save(self, filename):
        """
        Send the save request.

        Args:
            filename: the filename of the checkpoint
        """
        self.save_load_conn.req(('save', filename))

    def listen(self):
        """
        The listener for collecting the computation requests and performing neural network evaluation.
        """
        printlog('create network')
        self.net = network.Network(game_config, num_gpu=self.num_gpu,
                                   cluster=self.cluster, job=self.job, data_format='NCHW')
        if self.load_path is not None:
            printlog('load model')
            self.net.load(self.load_path)

        thrd.Thread(target=self.sl_listen, name='save_load_listener', daemon=True).start()

        printlog('loop begin')
        while True:
            try:
                reqs = []
                for i in range(self.max_batch_size):
                    block = i < self.num_gpu
                    reqs.append(self.server_client_conn.get(block))
            except EmptyExc:
                pass
            finally:
                # printlog(len(reqs), 'reqs')
                states_np = np.concatenate([req[0] for req in reqs], 0)
                rp, rv = self.net.response((states_np,))
                for i in range(len(reqs)):
                    reqs[i][1].send((rp[i], rv[i]))
