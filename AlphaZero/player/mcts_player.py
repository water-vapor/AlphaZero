import AlphaZero.search.mcts as MCTS

class Player:

    def __init__(self, eval_fun, game_config, ext_config):
        """
        Create MCT. MCT will be reused.
        Data consistency is NOT guaranteed because GameState of certain game is associate with this MCT but this class
        doesn't have the GameState. So this player should first think about a move (if it is its turn) and both players
        should acknowledge this move and update their MCT. And this class does NOT update the GameState.
        e.g.
            move = player_1.think(state)
            state.do_move(move)
            player_1.ack(move)
            player_2.ack(move)
        :param eval_fun: NNEvaluator class.
        """

        self._game_config = game_config
        self.mcts = MCTS.MCTSearch(eval_fun.eval, self._game_config, max_playout=ext_config['max_playout'])

    def think(self, state, dirichlet=False):
        move, probs = self.mcts.calc_move_with_probs(state, dirichlet)
        return move, probs

    def ack(self, move):
        self.mcts.update_with_move(move)
