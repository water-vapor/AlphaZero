import multiprocessing as mp
import atexit
import traceback as tb

import AlphaGoZero.game.player as player
import AlphaGoZero.game.nn_eval as nn_eval
import AlphaGoZero.game.gameplay as gameplay
import AlphaGoZero.go as go

def kill_children():
    for p in mp.active_children():
        p.terminate()

class Optimizer:
    def __init__(self, net, s_conn, data_queue, **kwargs):
        print('create optimizer')
        self.net = net
        self.s_conn = s_conn
        self.data_queue = data_queue
        self.num_ckpt = kwargs.get('num_ckpt', 1000)
        self.num_steps = kwargs.get('num_steps', 700000) # TODO: find correct steps
        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run)

    def __enter__(self):
        # self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def run(self):
        print('optimizer: training loop begin')
        for step in range(self.num_steps):
            data = self.data_queue.get()
            self.net.update(data)
            if (step + 1) % self.num_ckpt == 0:
                self.net.save('challenger_name') # TODO: use proper model name
                self.s_conn.send('challenger_name')