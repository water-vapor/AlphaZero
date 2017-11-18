import pickle
from AlphaGoZero.reinforcement_learning.sequential.match import Match


def selfplay(best_player_path, path_to_save, num_games=25000):
    """ Generate self play data. Search probabilities are stored in pickle files.

    Args:
        best_player_path: path to the best player
        path_to_save: path to save sgf and pkl files
        num_games: number of games to play

    Returns:
        None

    """
    # This can be parallelized
    match = Match(best_player_path, best_player_path)

    for num_game in range(num_games):
        _, _, search_probs = match.play(path_to_save, str(num_game) + '.sgf', is_selfplay=True)
        # Save the raw list to a pickle file
        with open(str(num_game) + '.pkl', 'rb+') as f:
            pickle.dump(search_probs, f)

        # TODO: auto resignation should be implemented
