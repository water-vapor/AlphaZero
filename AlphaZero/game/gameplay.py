import importlib

import numpy as np
import yaml

import AlphaZero.game.player as player
from AlphaZero.train.parallel.util import *


class Game:

    def __init__(self, nn_eval_1, nn_eval_2, game_config):
        """
        Set NN evaluator and game board.
        :param nn_eval_1: NNEvaluator class. This class doesn't create evaluator.
        :param nn_eval_2: NNEvaluator class.
        """
        self.player_1 = player.Player(nn_eval_1, game_config)
        self.player_2 = player.Player(nn_eval_2, game_config)
        self._game_env = importlib.import_module(game_config['env_path'])
        self.state = self._game_env.GameState()
        self.winner = None
        self.state_history = []
        self.probs_history = []
        self._w = game_config['board_width']
        self._h = game_config['board_height']
        self._f = game_config['history_step'] * game_config['planes_per_step'] + game_config['additional_planes']
        self._o = game_config['flat_move_output']
        self._preproc = importlib.import_module(game_config['state_converter_path'])
        self._state_tensor_converter = self._preproc.StateTensorConverter(game_config)

        with open('AlphaZero/config/gameplay.yaml') as f:
            ext_config = yaml.load(f)
        self.dirichlet_before = ext_config['dirichlet_before']
        self.log_iter = ext_config['log_iter']

    def start(self):
        """ Make the instance callable. Start playing.

        Returns: Game winner. Definition is in go.py.
        """
        current_player = self.player_1
        while not self.state.is_end_of_game:
            if self.state.turns % self.log_iter == 0:
                printlog(str(self.state.turns), 'moves')

            move, probs = current_player.think(self.state, (self.state.turns <= self.dirichlet_before))  # TODO: dirichlet doesn't work
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

            # TODO: other end game condition

        self.winner = self.state.get_winner()
        printlog('end', self.state.stones_played)
        return self.winner

    def get_history(self):
        # TODO: whether to put the whole game history in one batch
        state_np = np.zeros((len(self.state_history), self._f, self._w, self._h))
        probs_np = np.zeros((len(self.probs_history), self._o))
        result_np = np.zeros((len(self.probs_history)))
        for i in range(len(self.probs_history)):
            state_np[i] = self._state_tensor_converter.state_to_tensor(self.state_history[i])[0]
            for prob in self.probs_history[i]:
                # flat move will include PASS MOVE if applicable, since PASS will be of index w*h,
                # there will be no out of bound error
                if prob[0] == self._game_env.PASS_MOVE:
                    probs_np[i, self._w * self._h] = prob[1]
                else:
                    probs_np[i, prob[0][0] * self._w + prob[0][1]] = prob[1]
            result_np[i] = (i % 2 == (self.winner != self._game_env.BLACK))
        return state_np, probs_np, result_np

    # TODO: save state to sgf
