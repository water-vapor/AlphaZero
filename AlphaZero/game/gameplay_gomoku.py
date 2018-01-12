import AlphaZero.game.player as player
import AlphaZero.env.gomoku as gomoku
import AlphaZero.processing.gomoku.state_converter as preproc
from AlphaZero.train.parallel.util import *
import yaml
import os

import numpy as np
import importlib

go_config_path = os.path.join('AlphaZero', 'config', 'gomoku.yaml')
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
        self.state = gomoku.GameState()
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
        :return: Game winner. Definition is in gomoku.py.
        """
        current_player = self.player_1
        while not self.state.is_end_of_game:
            if self.state.stones_played % 30 == 0:
                printlog(str(self.state.stones_played), 'moves')

            move, probs = current_player.think(self.state)
            self.state_history.append(self.state.copy())
            self.probs_history.append(probs)
            self.state.do_move(move)
            self.player_1.ack(move)
            self.player_2.ack(move)

            # change player
            if current_player == self.player_1:
                current_player = self.player_2
            else:
                current_player = self.player_1

        self.winner = self.state.get_winner()
        printlog('end', self.state.stones_played)
        return self.winner

    def get_history(self):
        # TODO: whether to put the whole game history in one batch

        state_np = np.zeros((len(self.state_history), self._f, self._w, self._h))
        probs_np = np.zeros((len(self.probs_history), self._o))
        result_np = np.zeros((len(self.probs_history)))
        for i in range(len(self.probs_history)):
            state_np[i] = preproc.StateTensorConverter().state_to_tensor(self.state_history[i])[0]
            for prob in self.probs_history[i]:
                probs_np[i, prob[0][0] * self._w + prob[0][1]] = prob[1]
            result_np[i] = (i % 2 == (self.winner != gomoku.BLACK))
        return state_np, probs_np, result_np

    # TODO: save state to sgf
