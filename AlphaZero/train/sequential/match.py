import os
import random

from AlphaZero.env.go import GameState, BLACK, WHITE
from AlphaZero.math_helper import random_state_transform
from AlphaZero.mcts import MCTSearch
from AlphaZero.network.main import Network
from AlphaZero.util import save_gamestate_to_sgf


class Match(object):
    """ A class that plays the game between two NN players.
        Note that there has to be two sessions, because two players are mostly different.
        To parallelize, consider multiple matches sharing the player1's and player2's model session.
    """

    def __init__(self, player1, player2, base_dir='data', board_size=19, max_moves=400, num_gpu=1):
        """
            Arguments:
                player1: name of player 1
                player2: name of player 2
        """
        model_dir = os.path.join(base_dir, 'models')
        player1_path = os.path.join(model_dir, player1)
        player2_path = os.path.join(model_dir, player2)

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
        # player 1 refers to mcts[0]
        if player1_is_black:
            current_player = 0
        else:
            current_player = 1

        # Create a list to store search probabilities for self-play
        # if is_selfplay is not necessary, creating a list won't have side effects
        search_probs_history = []

        # Play the game
        for turn in range(self.max_moves):
            # Whether to enable exploration depends on the mode,
            # exploration is only enabled in the first 30 turns in a self-play
            if is_selfplay and turn + 1 <= 30:
                dirichlet = True
            else:
                dirichlet = False

            # Record search probabilities for self-play
            if is_selfplay:
                move, search_probs = mcts[current_player].calc_move_with_probs(gs, dirichlet)
                search_probs_history.append(search_probs)
            else:
                move = mcts[current_player].calc_move(gs, dirichlet)

            # Make the move and update both player's search tree
            # TODO: Should we use a single search tree for self-play?
            gs.do_move(move)
            for p in range(2):
                mcts[p].update_with_move(gs)

            # Toggle player
            current_player = 1 - current_player

            # Check if the game ends
            if gs.is_end_of_game:
                break

        # Game ends
        result = gs.get_winner()
        # TODO: Detailed result info, i.e. W+Resign
        if result is WHITE:
            result_string = "W+"
        elif result is BLACK:
            result_string = "B+"
        else:
            # TODO: How should we deal with ties? discarding ties for now
            return 0
        save_gamestate_to_sgf(gs, save_path, save_filename + '.sgf', result=result_string)

        # Return an extra list of search probabilities in the same order, other features can be extracted in sgf
        # search_probs_history is a list of a list of (action, probs)
        if is_selfplay:
            return player1_is_black, result, search_probs_history
        else:
            return player1_is_black, result
