from AlphaGoZero.Network.main import Model
from AlphaGoZero.util import save_gamestate_to_sgf
from AlphaGoZero.mcts import MCTSearch
from AlphaGoZero.math_helper import random_state_transform
from AlphaGoZero.go import GameState, PASS

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
		self.player1 = Model(num_gpu)
		self.player1.load(player1_path)

		self.player2 = Model(num_gpu)
		self.player2.load(player2_path)

	def play(self, save_path, save_filename):
		""" Plays the game and saves to a sgf file.
		"""

		mcts1 = MCTSearch(random_state_transform, lambda s: self.player1.response([s]))
		mcts2 = MCTSearch(random_state_transform, lambda s: self.player2.response([s]))
		mcts = [mcts1, mcts2]
		gs = GameState(size=board_size)

		# Randomly assign colors
		if np.random.randint(2) == 0:
			current_player = 1
		else:
			current_player = 0
		for _ in range(max_moves):
			move = mcts[current_player].calc_move(gs)
			gs.do_move(move)
			for p in range(2):
				mcts[p].update_with_move(gs)
			current_player = 1 - current_player
			# Check all possible exit conditions
			history = gs.get_history()
			if len(history) >= 2 and history[-1] is PASS and history[-2] is PASS:
				break
		save_gamestate_to_sgf(gs, save_path, save_filename)


