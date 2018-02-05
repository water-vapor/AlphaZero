import atexit
import traceback as tb

import numpy as np
import yaml

import AlphaZero.network.main as network
from AlphaZero.train.parallel.util import *


def kill_children():
    for p in mp.active_children():
        p.terminate()


class Optimizer:
    def __init__(self, cluster, job, s_conn, data_queue, game_config, **kwargs):
        printlog('create optimizer')
        self.cluster = cluster
        self.job = job
        self.net = None
        self.s_conn = s_conn
        self.data_queue = data_queue

        with open('AlphaZero/train/parallel/sys_config.yaml') as f:
            ext_config = yaml.load(f)['optimizer']
        self.num_ckpt = ext_config['num_ckpt']
        self.num_steps = ext_config['num_steps']
        self.batch_size = ext_config['batch_size']

        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run, name='opti')

        self.game_config = game_config

    def __enter__(self):
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def run(self):
        self.net = network.Network(self.game_config, config_file="AlphaZero/network/reinforce.yaml",
                                   cluster=self.cluster, job=self.job)

        self.data_queue.start_training.acquire()
        printlog('optimizer: training loop begin')
        for step in range(self.num_steps):
            data = self.data_queue.get(self.batch_size)
            self.net.update(data)
            if step % 30 == 0:
                printlog('update iter', step)
            if (step + 1) % self.num_ckpt == 0:
                self.net.save('./' + self.game_config['name'] + '_model/ckpt')  # TODO: use proper model name
                self.s_conn.send(step + 1)


class Datapool:
    def __init__(self):
        self.data_pool = None
        self.start_training = mp.Semaphore(0)

        with open('AlphaZero/train/parallel/sys_config.yaml') as f:
            ext_config = yaml.load(f)['datapool']
        self.pool_size = ext_config['pool_size']
        self.start_data_size = ext_config['start_data_size']

        conn_num = ext_config['conn_num']
        self.rcpt = Reception(conn_num)

        self.server = mp.Process(target=self.serve, name='data_pool_server')

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.server.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def serve(self):
        printlog('start')
        while True:
            (value, req_type), s_conn = self.rcpt.get()
            if req_type == 'put':
                # printlog('get packet')
                data = value
                if self.data_pool is None:
                    printlog('init pool')
                    self.data_pool = [item for item in data]
                else:
                    printlog('add data to pool')
                    self.data_pool = [np.concatenate([it, it_new], axis=0) for it, it_new in zip(self.data_pool, data)]
                    if self.data_pool[0].shape[0] > self.pool_size:
                        printlog('delete old data')
                        self.data_pool = [it[-self.pool_size:] for it in self.data_pool]
                if self.data_pool[0].shape[0] > self.start_data_size:
                    self.start_training.release()
                s_conn.send('done')
            elif req_type == 'get':
                # printlog('sample data')
                batch_size = value
                idxs = np.random.choice(range(self.data_pool[0].shape[0]), batch_size)
                data = (self.data_pool[0][idxs], self.data_pool[1][idxs], self.data_pool[2][idxs])
                s_conn.send(data)

    def put(self, data):
        self.rcpt.req((data, 'put'))

    def get(self, batch_size):
        data = self.rcpt.req((batch_size, 'get'))
        return data
