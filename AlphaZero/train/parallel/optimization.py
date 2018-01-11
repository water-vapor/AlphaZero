import atexit
import traceback as tb
import numpy as np

from AlphaZero.train.parallel.util import *
import AlphaZero.network.main as network


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
        self.num_ckpt = kwargs.get('num_ckpt', 100)
        self.num_steps = kwargs.get('num_steps', 700000)  # TODO: find correct steps
        self.batch_size = kwargs.get('batch_size', 8)

        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run, name='opti')

        self._game_config = game_config

    def __enter__(self):
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def run(self):
        self.net = network.Network(self._game_config, config_file="AlphaZero/network/reinforce.yaml", cluster=self.cluster, job=self.job)

        self.data_queue.start_training.acquire()
        printlog('optimizer: training loop begin')
        for step in range(self.num_steps):
            data = self.data_queue.get(self.batch_size)
            self.net.update(data)
            if step % 30 == 0:
                printlog('update iter', step)
            if (step + 1) % self.num_ckpt == 0:
                self.net.save('./model/ckpt')  # TODO: use proper model name
                self.s_conn.send(step + 1)


class Datapool:
    def __init__(self, pool_size, start_data_size):
        self.data_pool = None
        self.start_training = mp.Semaphore(0)
        self.pool_size = pool_size
        self.start_data_size = start_data_size

        conn_num = 20  # TODO: use proper number
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
