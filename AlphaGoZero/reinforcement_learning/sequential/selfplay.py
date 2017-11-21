import pickle, os
from AlphaGoZero.reinforcement_learning.sequential.match import Match


def selfplay(best_player, base_dir='data', num_games=25000):
    """ Generate self play data and search probabilities. 
		Results are stored in data/selfplay/<best_player_name>/
		Game records are stored as sgf files.
		Search probabilities are stored as pickle files.

    Args:
        best_player: the name of the best player
        num_games: number of games to play

    Returns:
        None

    """
    # This can be parallelized
    match = Match(best_player, best_player, base_dir=base_dir)

    for num_game in range(num_games):
        _, _, search_probs = match.play(os.path.join(base_dir, 'selfplay', best_player), str(num_game) + '.sgf', is_selfplay=True)
        # Save the raw list to a pickle file
        with open(os.path.join(base_dir, 'selfplay', best_player, str(num_game) + '.pkl'), 'rb+') as f:
            pickle.dump(search_probs, f)

        # TODO: auto resignation should be implemented
