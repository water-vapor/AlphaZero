class Player:

    def __init__(self, nn_eval, game_config):
        """
        """

        self._game_config = game_config
        self.eval_fun = nn_eval.eval

    def think(self, state):
        move_candidates, _ = self.eval_fun(state.copy())
        moves = [(action, prob) for action, prob in move_candidates if state.is_legal(action)]
        move, _ = max(moves, key=lambda mv: mv[1])
        return move, None

    def ack(self, move):
        pass
