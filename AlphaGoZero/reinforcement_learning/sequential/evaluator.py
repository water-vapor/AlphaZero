import os
import random
import numpy as np
from AlphaGoZero.game.nn_eval import NNEvaluator
from AlphaGoZero.game.gameplay import Game
from AlphaGoZero.go import BLACK, WHITE


def evaluate(best_player_name, player_to_eval_name, base_dir='data', num_games=400, num_gpu=1):
    """ Plays games between best and the current player, returns the result.
        All game records are saved to data/evaluations/, though it is not used in the current RL training pipeline.

        Arguments:
            best_player_name: name of the best player's model
            player_to_eval_name: name of the player's model to evaluate against
        Returns:
            best_player_defeated: whether the best player lost 55% of the games
    """

    best_player = NNEvaluator(os.path.join(base_dir, 'models', best_player_name))
    player_to_eval = NNEvaluator(os.path.join(base_dir, 'models', player_to_eval_name))
    if random.randint(0, 1) == 0:
        black_player, white_player = best_player, player_to_eval
    else:
        black_player, white_player = player_to_eval, best_player
    match = Game(black_player, white_player)
    best_win = 0
    for num_game in range(num_games):
        result = match.start()
        with open(os.path.join(base_dir, 'evaluations',
                               best_player_name + '_vs_' + player_to_eval, str(num_game) + '.npy'), 'wb+') as f:
            np.save(f, match.get_history())
        if (black_player is best_player and result is BLACK) or (white_player is best_player and result is WHITE):
            best_win += 1

    if best_win / num_games <= 0.45:
        return True
    else:
        return False
