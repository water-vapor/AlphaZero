import AlphaGoZero.mcts as MCTS
import AlphaGoZero.math_helper as helper

def get_move_single(state, nn_eval):
    mcts = MCTS.MCTSearch(helper.random_state_transform, nn_eval)
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
        self.mcts = MCTS.MCTSearch(helper.random_state_transform, nn_eval)

    def think(self, state):
        move = self.mcts.calc_move(state)
        return move

    def observe(self, move):
        self.mcts.update_with_move(move)
