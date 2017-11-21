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

class Evaluator:

    def __init__(self, nn_eval_chal, nn_eval_best, r_conn, s_conn, num_games=400):
        print('create evaluator')
        self.num_games = num_games
        self.nn_eval_chal = nn_eval_chal
        self.nn_eval_best = nn_eval_best
        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run)
        self.r_conn, self.s_conn = r_conn, s_conn
        self.wait_r = mp.Semaphore(0)
        self.counter_lock = mp.Lock()
        self.win_counter = 0

        self.worker_lim = mp.Semaphore(mp.cpu_count()) # TODO: worker number

    def __enter__(self):
        print('evaluator: start proc')
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('evaluator: terminate proc')
        self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def count(self):
        self.counter_lock.acquire()
        self.win_counter += 1
        self.counter_lock.release()

    def eval_wrapper(self, color_of_new):
        # This block of code serves as a gate. This thread will not block updater but updater can block this thread
        self.nn_eval_chal.loading.acquire()            # nn_eval_new is not loading
        self.nn_eval_best.loading.acquire()           # nn_eval_best is not loading
        self.nn_eval_best.loading.release()           # release lock
        self.nn_eval_chal.loading.release()            # release lock

        self.nn_eval_chal.active_game.acquire(False)   # decrement counter of nn_eval_new
        self.nn_eval_best.active_game.acquire(False)  # decrement counter of nn_eval_best

        game = gameplay.Game(self.nn_eval_chal, self.nn_eval_best) if color_of_new == go.BLACK else gameplay.Game(self.nn_eval_best, self.nn_eval_chal)
        winner = game.start()
        if winner == color_of_new:
            self.count()

        self.nn_eval_best.active_game.release()       # increment counter
        self.nn_eval_chal.active_game.release()        # increment counter

        self.worker_lim.release()

    def run(self):
        print('evaluator: loop begin')
        while True:
            new_model_path = self.r_conn.recv()
            # update Network
            self.nn_eval_chal.load(new_model_path)
            self.win_counter = 0
            # open pool
            color_of_new_list = [go.BLACK, go.WHITE]*(self.num_games//2) + [go.BLACK]*(self.num_games%2)
            for c in color_of_new_list:
                self.worker_lim.acquire()
                mp.Process(target=self.eval_wrapper, args=(c,)).start()
            # wait
            if self.win_counter >= int(0.55 * self.num_games):
                # save model
                self.nn_eval_chal.save('best_name') # TODO: use proper model name
                # send path
                self.s_conn.send('best_name')
