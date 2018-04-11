import atexit
import importlib
import traceback as tb
import yaml
import argparse
import time
import os

import tensorflow as tf

# import AlphaZero.game.go.gameplay as gameplay
from AlphaZero.train.parallel.util import *
import AlphaZero.game.nn_eval as nn_eval

with open('AlphaZero/config/game.yaml') as f:
    game_selection = yaml.load(f)['game']
with open('AlphaZero/config/' + game_selection + '.yaml') as c:
    game_config = yaml.load(c)
_gameplay = importlib.import_module(game_config['gameplay_path'])


def kill_children():
    for p in mp.active_children():
        p.terminate()


class Selfplay:
    def __init__(self, nn_eval, r_conn, data_queue, game_config, ext_config):
        printlog('create selfplay')

        self.nn_eval = nn_eval
        self.r_conn = r_conn
        self.data_queue = data_queue
        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run, name='selfplay_game_launcher')
        self.listen_proc = mp.Process(target=self.listen_update, name='selfplay_listener')
        self.num_worker = ext_config['num_worker']
        self.worker_lim = mp.Semaphore(self.num_worker)
        self.game_config = game_config
        self.ext_config = ext_config

    def __enter__(self):
        printlog('selfplay: start proc')
        self.proc.start()
        self.listen_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        printlog('selfplay: terminate proc')
        self.proc.terminate()
        self.listen_proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def selfplay_wrapper(self):
        # process comm
        self.nn_eval.rwlock.r_acquire()
        # start game
        game = _gameplay.Game(self.nn_eval, self.nn_eval, self.game_config, self.ext_config['gameplay'])
        game.start()
        # get game history
        # convert
        data = game.get_history()
        # TODO: random flip
        # put in queue
        self.data_queue.put(data)
        # process comm
        self.nn_eval.rwlock.r_release()

        self.worker_lim.release()

    def run(self):
        printlog('start')
        cnt = 0
        while True:
            self.worker_lim.acquire()
            mp.Process(target=self.selfplay_wrapper,
                       name=self.game_config['name'] + '_selfplay_game_' + str(cnt)).start()
            cnt += 1

    def listen_update(self):
        printlog('listening')
        while True:
            path = self.r_conn.recv()
            self.nn_eval.load('./' + self.game_config['name'] + '_model/ckpt-' + str(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generator for trainer.')
    parser.add_argument('addr', type=str, help='master address')
    args = parser.parse_args()

    with open('AlphaZero/config/game.yaml') as f:
        game_selection = yaml.load(f)['game']
    config_path = os.path.join('AlphaZero', 'config', game_selection + '.yaml')
    if not os.path.exists(config_path):
        raise NotImplementedError("{} game config file does not exist.".format(game_selection))
    else:
        with open(config_path) as c:
            game_config = yaml.load(c)
    with open('AlphaZero/config/rl_sys_config.yaml') as f:
        ext_config = yaml.load(f)

    cluster = tf.train.ClusterSpec({'best':['localhost:4335']})

    mp.freeze_support()

    eval_dgen_r, eval_dgen_s = Block_Pipe()
    dgen_opti_q = Remote_Queue(args.addr, ext_config['datapool']['remote_port'])

    with nn_eval.NNEvaluator(cluster, game_config, ext_config['best']) as nn_eval_best, \
         Selfplay(nn_eval_best, eval_dgen_r, dgen_opti_q, game_config, ext_config['selfplay']) as dgen:

        while True:
            time.sleep(30)