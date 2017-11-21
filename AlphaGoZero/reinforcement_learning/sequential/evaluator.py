import os
from AlphaGoZero.reinforcement_learning.sequential.match import Match
from AlphaGoZero.go import BLACK, WHITE


def evaluate(best_player, player_to_eval, base_dir='data', num_games=400, num_gpu=1):
    """ Plays games between best and the current player, returns the result.
		All game records are saved to data/evaluations/, though it is not used in the current RL training pipeline.
		
		Arguments:
			best_player: name of the best player's model
			player_to_eval: name of the player's model to evaluate against 
        Returns:
            best_player_defeated: whether the best player lost 55% of the games
    """
    # This should run in parallel
    match = Match(best_player, player_eval, base_dir=base_dir)
    best_win = 0
    for num_game in range(num_games):
        best_is_back, result = match.play(os.path.join(base_dir, 'evaluations', best_player + '_vs_' + player_to_eval), str(num_game))
        if (best_is_back and result is BLACK) or (not best_is_back and result is WHITE):
            best_win += 1

    if best_win / num_games <= 0.45:
        return True
    else:
        return False
