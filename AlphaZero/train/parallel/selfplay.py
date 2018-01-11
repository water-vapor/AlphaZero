import traceback as tb
import atexit
import importlib

#import AlphaZero.game.go.gameplay as gameplay
from AlphaZero.train.parallel.util import *


def kill_children():
    for p in mp.active_children():
        p.terminate()


class Selfplay:
    def __init__(self, nn_eval, r_conn, data_queue, game_config):
        printlog('create selfplay')
        self.nn_eval = nn_eval
        self.r_conn = r_conn
        self.data_queue = data_queue
        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run, name='selfplay_game_launcher')
        self.listen_proc = mp.Process(target=self.listen_update, name='selfplay_listener')
        self.worker_lim = mp.Semaphore(2)  # TODO: worker number
        self.game_config = game_config
        self._gameplay = importlib.import_module(game_config['gameplay_path'])

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
        game = self._gameplay.Game(self.nn_eval, self.nn_eval)
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
            mp.Process(target=self.selfplay_wrapper, name='selfplay_game_' + str(cnt)).start()
            cnt += 1

    def listen_update(self):
        printlog('listening')
        while True:
            path = self.r_conn.recv()
            self.nn_eval.load('./model/ckpt-' + str(path))
