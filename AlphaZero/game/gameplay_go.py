import AlphaZero.game.player as player
import AlphaZero.env.go as go
import AlphaZero.processing.go.state_converter as preproc
from AlphaZero.train.parallel.util import *
import yaml
import os

import numpy as np
import importlib

go_config_path = os.path.join('AlphaZero', 'config', 'go.yaml')
with open(go_config_path) as c:
    game_config = yaml.load(c)

class Game:

    def __init__(self, nn_eval_1, nn_eval_2):
        """
        Set NN evaluator and game board.
        :param nn_eval_1: NNEvaluator class. This class doesn't create evaluator.
        :param nn_eval_2: NNEvaluator class.
        """
        self.player_1 = player.Player(nn_eval_1, game_config)
        self.player_2 = player.Player(nn_eval_2, game_config)
        self.state = go.GameState()
        self.winner = None
        self.state_history = []
        self.probs_history = []
        self._w = game_config['board_width']
        self._h = game_config['board_height']
        self._f = game_config['history_step'] * game_config['planes_per_step'] + game_config['additional_planes']
        self._o = game_config['flat_move_output']

    def start(self):
        """
        Make the instance callable. Start playing.
        :param args:
        :param kwargs:
        :return: Game winner. Definition is in go.py.
        """
        pass_1 = pass_2 = False
        cnt = 0
        while not (pass_1 and pass_2):
            if cnt % 30 == 0:
                printlog(str(cnt), 'moves')

            move, probs = self.player_1.think(self.state)
            if move is go.PASS_MOVE:
                pass_1 = True  # TODO: check the type of variable move
            self.state_history.append(self.state.copy())
            self.probs_history.append(probs)
            self.state.do_move(move)
            self.player_1.ack(move)
            self.player_2.ack(move)
            cnt += 1

            move, probs = self.player_2.think(self.state)
            if move is go.PASS_MOVE:
                pass_2 = True
            self.state_history.append(self.state.copy())
            self.probs_history.append(probs)
            self.state.do_move(move)
            self.player_1.ack(move)
            self.player_2.ack(move)
            cnt += 1

            # TODO: other end game condition

        self.winner = self.state.get_winner()
        printlog('end', cnt)
        return self.winner

    def get_history(self):
        # TODO: whether to put the whole game history in one batch
        state_np = np.zeros((len(self.state_history), self._f, self._w, self._h))
        probs_np = np.zeros((len(self.probs_history), self._o))
        result_np = np.zeros((len(self.probs_history)))
        for i in range(len(self.probs_history)):
            state_np[i] = preproc.StateTensorConverter().state_to_tensor(self.state_history[i])[0]
            for prob in self.probs_history[i]:
                if prob[0] == go.PASS_MOVE:
                    probs_np[i, self._w * self._h] = prob[1]
                else:
                    probs_np[i, prob[0][0] * self._w + prob[0][1]] = prob[1]
            result_np[i] = (i % 2 == (self.winner != go.BLACK))
        return state_np, probs_np, result_np

    # TODO: save state to sgf
