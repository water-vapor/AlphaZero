import random
from AlphaGoZero.Network.main import Network
from AlphaGoZero.util import save_gamestate_to_sgf
from AlphaGoZero.mcts import MCTSearch
from AlphaGoZero.math_helper import random_state_transform
from AlphaGoZero.go import GameState, BLACK, WHITE


class Match(object):
    """ A class that plays the game between two NN players.
        Note that there has to be two sessions, because two players are mostly different.
        To parallelize, consider multiple matches sharing the player1's and player2's model session.
    """

    def __init__(self, player1_path, player2_path, board_size=19, max_moves=400, num_gpu=1):
        """
            Arguments:
                player1_path: the path to the model of player1
                player2_path: the path to the model of player2
        """
        # These models can be shared in parallel version
        self.player1 = Network(num_gpu)
        self.player1.load(player1_path)

        self.player2 = Network(num_gpu)
        self.player2.load(player2_path)

        self.board_size = board_size
        self.max_moves = max_moves

    def play(self, save_path, save_filename, is_selfplay=False):
        """ Plays the game and saves to a sgf file.
        """

        mcts1 = MCTSearch(random_state_transform, lambda s: self.player1.response([s]))
        mcts2 = MCTSearch(random_state_transform, lambda s: self.player2.response([s]))
        mcts = [mcts1, mcts2]
        gs = GameState(size=self.board_size)

        # Randomly assign colors
        player1_is_black = random.randint(0, 1) == 0
        if player1_is_black:
            current_player = 0
        else:
            current_player = 1
        # Play the game
        for turn in range(self.max_moves):
            # Whether to enable exploration depends on the mode
            if is_selfplay and turn+1 <= 30:
                dirichlet = True
            else:
                dirichlet = False
            move = mcts[current_player].calc_move(gs, dirichlet)
            gs.do_move(move)
            for p in range(2):
                mcts[p].update_with_move(gs)
            current_player = 1 - current_player
            # Check all possible exit conditions
            history = gs.get_history()
            if gs.is_end_of_game:
                break
        # Game ends
        result = gs.get_winner()
        if result is WHITE:
            result_string = "W"
        elif result is BLACK:
            result_string = "B"
        else:
            # TODO: How should we deal with ties? discarding ties for now
            return 0
        save_gamestate_to_sgf(gs, save_path, save_filename, result=result_string)
        return player1_is_black, result
