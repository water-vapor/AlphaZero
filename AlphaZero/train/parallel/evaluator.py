import atexit
import importlib
import traceback as tb
import yaml

# import AlphaZero.game.go.gameplay as gameplay
# import AlphaZero.env.go as go
from AlphaZero.train.parallel.util import *

with open('AlphaZero/config/game.yaml') as f:
    game_selection = yaml.load(f)['game']
with open('AlphaZero/config/' + game_selection + '.yaml') as c:
    game_config = yaml.load(c)
_game_env = importlib.import_module(game_config['env_path'])
_gameplay = importlib.import_module(game_config['gameplay_path'])


# Selection logic in evaluator only work in two-player games, and they must be named BLACK and WHITE in game env

def kill_children():
    for p in mp.active_children():
        p.terminate()


class Evaluator:

    def __init__(self, nn_eval_chal, nn_eval_best, r_conn, s_conn, game_config, ext_config):
        printlog('create evaluator')

        self.num_games = ext_config['num_games']
        self.nn_eval_chal = nn_eval_chal
        self.nn_eval_best = nn_eval_best
        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run, name='evaluator')
        self.r_conn, self.s_conn = r_conn, s_conn
        self.win_counter = mp.Value('i', 0)

        self.num_worker = ext_config['num_worker']
        self.worker_lim = mp.Semaphore(self.num_worker)

        self.join_worker = mp.Semaphore(0)
        self.finished_worker = mp.Value('i', 0)

        self.game_config = game_config
        self.ext_config = ext_config

    def __enter__(self):
        printlog('evaluator: start proc')
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        printlog('evaluator: terminate proc')
        self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def eval_wrapper(self, color_of_new):
        self.nn_eval_chal.rwlock.r_acquire()
        self.nn_eval_best.rwlock.r_acquire()

        # printlog('begin')
        if color_of_new == _game_env.BLACK:
            game = _gameplay.Game(self.nn_eval_chal, self.nn_eval_best, self.game_config, self.ext_config['gameplay'])
        else:
            game = _gameplay.Game(self.nn_eval_best, self.nn_eval_chal, self.game_config, self.ext_config['gameplay'])
        winner = game.start()
        if winner == color_of_new:
            self.win_counter.value += 1
        printlog('winner', winner)

        self.worker_lim.release()
        self.finished_worker.value += 1
        if self.finished_worker.value == self.num_games:
            self.join_worker.release()

        self.nn_eval_best.rwlock.r_release()  # increment counter
        self.nn_eval_chal.rwlock.r_release()  # increment counter

    def run(self):
        printlog('loop begin')
        while True:
            new_model_path = self.r_conn.recv()
            # update Network
            printlog('load network')
            self.nn_eval_chal.load('./' + self.game_config['name'] + '_model/ckpt-' + str(new_model_path))
            self.win_counter.value = 0
            self.finished_worker.value = 0
            # open pool
            color_of_new_list = [_game_env.BLACK, _game_env.WHITE] * (self.num_games // 2) + [
                _game_env.BLACK] * (self.num_games % 2)
            for i, c in enumerate(color_of_new_list):
                self.worker_lim.acquire()
                mp.Process(target=self.eval_wrapper, args=(c,), name='eval_game_' + str(i)).start()
            # wait
            self.join_worker.acquire()
            printlog('win rate', self.win_counter.value / self.num_games)
            if self.win_counter.value >= int(0.55 * self.num_games):
                # save model
                # self.nn_eval_chal.save('./model/best_name')
                # send path
                self.s_conn.send(new_model_path)
