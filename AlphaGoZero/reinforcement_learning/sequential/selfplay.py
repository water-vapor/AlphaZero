from AlphaGoZero.reinforcement_learning.sequential.match import Match


def selfplay(best_player_path, path_to_save, num_games=25000):
    # This can be parallelized
    match = Match(best_player_path, best_player_path)

    for num_game in range(num_games):
        match.play(path_to_save, str(num_game), is_selfplay=True)

        # TODO: auto resignation should be implemented
