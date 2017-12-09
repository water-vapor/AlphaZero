import AlphaGoZero.mcts as MCTS
import AlphaGoZero.math_helper as helper

def get_move_single(state, nn_eval):
    mcts = MCTS.MCTSearch(helper.random_state_transform, nn_eval.eval)
    move = mcts.calc_move(state)
    return move

def play_single(state, nn_eval):
    """
    play a single move
    :param state: GameState instance
    :param nn_eval: evaluation function
    :return: a *new copy* of state after move
    """
    state_new = state.copy()
    move = get_move_single(state, nn_eval)
    state_new.do_move(move)
    return state_new

class Player:

    def __init__(self, nn_eval):
        """
        Create MCT. MCT will be reused.
        Data consistency is NOT guaranteed because GameState of certain game is associate with this MCT but this class
        doesn't have the GameState. So this player should first think about a move (if it is its turn) and both players
        should acknowledge this move and update their MCT. And this class does NOT update the GameState.
        e.g.
            move = player_1.think(state)
            state.do_move(move)
            player_1.observe(move)
            player_2.observe(move)
        :param nn_eval: NNEvaluator class.
        """
        self.mcts = MCTS.MCTSearch(helper.random_state_transform, nn_eval.eval, max_playout=5)

    def think(self, state):
        # TODO: add dirichlet noise for first 30 moves
        move, probs = self.mcts.calc_move_with_probs(state)
        return move, probs

    def ack(self, move):
        self.mcts.update_with_move(move)
