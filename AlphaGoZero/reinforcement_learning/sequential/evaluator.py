from AlphaGoZero.reinforcement_learning.sequential.match import Match
from AlphaGoZero.go import BLACK, WHITE


def evaluate(best_player_path, player_eval_path, save_path='tmp/', num_games=400, num_gpu=1):
    """
        Returns:
            best_player_defeated: the best player lost 55% of the games, it should be replaced
    """
    # This should run in parallel
    match = Match(best_player_path, player_eval_path)
    best_win = 0
    for num_game in range(num_games):
        best_is_back, result = match.play(save_path, str(num_game))
        if (best_is_back and result is BLACK) or (not best_is_back and result is WHITE):
            best_win += 1

    if best_win / num_games <= 0.45:
        return True
    else:
        return False
