import atexit
import importlib
import traceback as tb
import yaml
import argparse
import time
import os
import multiprocessing as mp
import threading as thrd
import tensorflow as tf

# import AlphaZero.game.go.gameplay as gameplay
from AlphaZero.train.parallel.util import *
import AlphaZero.evaluator.nn_eval_parallel as nn_eval

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
        self.num_worker = ext_config['num_worker']
        self.remote_port = ext_config['remote_port']
        self.remote_update_port = ext_config['remote_update_port']
        self.remote_worker_reg = {}
        self.worker_lim = mp.Semaphore(self.num_worker)
        self.game_config = game_config
        self.ext_config = ext_config

    def __enter__(self):
        printlog('selfplay: start proc')
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        printlog('selfplay: terminate proc')
        self.proc.terminate()
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

        thrd.Thread(target=self.listen_update, name='selfplay_listener', daemon=True).start()
        if self.ext_config.get('remote'):
            thrd.Thread(target=self.remote_listen_update, name='selfplay_remote_update', daemon=True).start()
        else:
            thrd.Thread(target=self.remote_rcv, name="selfplay_remote_rcv", daemon=True).start()

        cnt = 0
        while True:
            self.worker_lim.acquire()
            mp.Process(target=self.selfplay_wrapper,
                       name=self.game_config['name'] + '_selfplay_game_' + str(cnt)).start()
            cnt += 1

    def listen_update(self):
        printlog_thrd('listening')
        while True:
            path = self.r_conn.recv()
            for addr, _ in self.remote_worker_reg.items():
                printlog_thrd('remote update', addr)
                remote_q = Remote_Queue(addr, self.remote_update_port)
                remote_q.put('./' + self.game_config['name'] + '_model/ckpt-' + str(path))
            self.nn_eval.load('./' + self.game_config['name'] + '_model/ckpt-' + str(path))

    def remote_rcv(self):
        server_socket = socket.socket()
        server_socket.bind(('', self.remote_port))

        while True:
            server_socket.listen(1)
            printlog_thrd('waiting for a connection...')
            client_connection, client_address = server_socket.accept()
            printlog_thrd('connected to', client_address[0])
            ultimate_buffer = b''
            while True:
                receiving_buffer = client_connection.recv(2 ** 20)
                if not len(receiving_buffer): break
                ultimate_buffer += receiving_buffer
            final_image = pickle.loads(ultimate_buffer)
            client_connection.close()
            printlog_thrd('frame received')
            self.data_queue.put(final_image)
            self.remote_worker_reg[client_address[0]] = True

    def remote_listen_update(self):
        server_socket = socket.socket()
        server_socket.bind(('', self.remote_update_port))

        while True:
            server_socket.listen(1)
            printlog_thrd('waiting for a connection...')
            client_connection, client_address = server_socket.accept()
            printlog_thrd('connected to', client_address[0])
            ultimate_buffer = b''
            while True:
                receiving_buffer = client_connection.recv(2 ** 20)
                if not len(receiving_buffer): break
                ultimate_buffer += receiving_buffer
            final_image = pickle.loads(ultimate_buffer)
            client_connection.close()
            printlog_thrd('frame received')
            self.nn_eval.load(final_image)


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

    cluster = tf.train.ClusterSpec({'best': ['localhost:4335']})
    ext_config['selfplay']['remote'] = True

    mp.freeze_support()

    eval_dgen_r, eval_dgen_s = Block_Pipe()
    dgen_opti_q = Remote_Queue(args.addr, ext_config['selfplay']['remote_port'])

    with nn_eval.NNEvaluator(cluster, game_config, ext_config['best']) as nn_eval_best, \
            Selfplay(nn_eval_best, eval_dgen_r, dgen_opti_q, game_config, ext_config['selfplay']) as dgen:

        while True:
            time.sleep(30)
