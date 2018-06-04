class Player:
    """
    Represents a player playing according to an evaluation function.
    """

    def __init__(self, nn_eval, game_config):
        """

        Args:
            nn_eval: neural network evaluation function.
            game_config: game config file.
        """

        self._game_config = game_config
        self.eval_fun = nn_eval.eval

    def think(self, state):
        """
        Chooses the move with the highest probability by evaluating the current state with the evaluation function.
        Args:
            state: the current game state.

        Returns:
            tuple: a tuple of the calculated move and None.
        """
        move_candidates, _ = self.eval_fun(state.copy())
        moves = [(action, prob) for action, prob in move_candidates if state.is_legal(action)]
        move, _ = max(moves, key=lambda mv: mv[1])
        return move, None

    def ack(self, move):
        """
        Does nothing.

        Args:
            move: the current move.

        Returns:
            None
        """
        pass
