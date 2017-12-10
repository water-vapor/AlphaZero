import multiprocessing as mp
import atexit
import traceback as tb

import AlphaGoZero.game.player as player
import AlphaGoZero.game.nn_eval as nn_eval
import AlphaGoZero.game.gameplay as gameplay
import AlphaGoZero.go as go
from AlphaGoZero.reinforcement_learning.parallel.util import *

def kill_children():
    for p in mp.active_children():
        p.terminate()

class Evaluator:

    def __init__(self, nn_eval_chal, nn_eval_best, r_conn, s_conn, num_games=10):
        printlog('create evaluator')
        self.num_games = num_games
        self.nn_eval_chal = nn_eval_chal
        self.nn_eval_best = nn_eval_best
        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run, name='evaluator')
        self.r_conn, self.s_conn = r_conn, s_conn
        self.wait_r = mp.Semaphore(0)
        self.win_counter = mp.Value('i', 0)

        self.num_worker = 2
        self.worker_lim = mp.Semaphore(self.num_worker) # TODO: worker number

        self.join_worker = mp.Semaphore(0)
        self.finished_worker = mp.Value('i', 0)

    def __enter__(self):
        printlog('evaluator: start proc')
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        printlog('evaluator: terminate proc')
        self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def count(self):
        self.win_counter.value += 1

    def eval_wrapper(self, color_of_new):
        self.nn_eval_chal.rwlock.r_acquire()
        self.nn_eval_best.rwlock.r_acquire()

        printlog('begin')
        game = gameplay.Game(self.nn_eval_chal, self.nn_eval_best) if color_of_new == go.BLACK else gameplay.Game(self.nn_eval_best, self.nn_eval_chal)
        winner = game.start()
        if winner == color_of_new:
            self.count()
        printlog('winner', winner)

        self.worker_lim.release()
        self.finished_worker.value += 1
        if self.finished_worker.value == self.num_games:
            self.join_worker.release()

        self.nn_eval_best.rwlock.r_release()       # increment counter
        self.nn_eval_chal.rwlock.r_release()        # increment counter

    def run(self):
        printlog('loop begin')
        while True:
            new_model_path = self.r_conn.recv()
            # update Network
            printlog('load network')
            self.nn_eval_chal.load('./model/ckpt-' + str(new_model_path))
            self.win_counter.value = 0
            # open pool
            color_of_new_list = [go.BLACK, go.WHITE]*(self.num_games//2) + [go.BLACK]*(self.num_games%2)
            for i, c in enumerate(color_of_new_list):
                self.worker_lim.acquire()
                mp.Process(target=self.eval_wrapper, args=(c,), name='eval_game_'+str(i)).start()
            # wait
            self.join_worker.acquire()
            printlog('win rate', self.win_counter.value/self.num_games)
            if self.win_counter.value >= int(0.55 * self.num_games):
                # save model
                # self.nn_eval_chal.save('./model/best_name') # TODO: use proper model name
                # send path
                self.s_conn.send(new_model_path)