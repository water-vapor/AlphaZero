import multiprocessing as mp
import traceback as tb
import atexit
import random
import time

import AlphaGoZero.go as go
import AlphaGoZero.game.gameplay as gameplay

def kill_children():
    for p in mp.active_children():
        p.terminate()

class Selfplay:
    def __init__(self, nn_eval, r_conn, data_queue):
        print('create selfplay')
        self.nn_eval = nn_eval
        self.r_conn = r_conn
        self.data_queue = data_queue
        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run)
        self.listen_proc = mp.Process(target=self.listen_update)
        self.worker_lim = mp.Semaphore(mp.cpu_count()) # TODO: worker number

    def __enter__(self):
        print('selfplay: start proc')
        self.proc.start()
        self.listen_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('selfplay: terminate proc')
        self.proc.terminate()
        self.listen_proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def selfplay_wrapper(self):
        # process comm
        self.nn_eval.loading.acquire()
        self.nn_eval.loading.release()

        self.nn_eval.active_game.acquire(False)
        # start game
        print('selfplay game: begin')
        game = gameplay.Game(self.nn_eval, self.nn_eval)
        game.start()
        print('selfplay game: end')
        # get game history
        # convert
        data = game.get_history()
        # put in queue
        self.data_queue.put(data)
        # process comm
        self.nn_eval.active_game.release()

        self.worker_lim.release()

    def run(self):
        print('selfplay: create pool')
        while True:
            self.worker_lim.acquire()
            mp.Process(target=self.selfplay_wrapper).start()

    def listen_update(self):
        print('selfplay update: listening')
        while True:
            path = self.r_conn.recv()
            self.nn_eval.load(path)
