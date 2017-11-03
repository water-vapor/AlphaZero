import numpy as np
import math
import AlphaGoZero.settings as s


class MCTreeNode(object):
	"""Tree Node in MCTS.
	"""

	def __init__(self, parent, prior_prob):

		self._parent = parent
		self._children = {}
		# N(s,a)
		self._visit_cnt = 0
		# W(s,a)
		self._total_action_val = 0
		# Q(s,a)
		self._mean_action_val = 0
		# P(s,a)
		self._prior_prob = prior_prob

	def expand(self, policy, value):
		""" Expand a leaf node according to the network evaluation.
		Arguments:
			policy: a list of (action, prob) tuples returned by the network
			value: the value of this node returned by the network
		"""
		# Check if the node is leaf
		if not self.is_leaf():
			return

		# TODO: Should value be inverted according to the player?
		# Update W(s,a) for this parent node by formula W(s,a) = W(s,a) + v
		self._total_action_val += value
		# Create valid children
		for action, prob in policy:
			# TODO: Is there an extra condition to check pass and suicide?
			#		Assuming illegal move is removed from nn_output already.
			if action not in self._children:
				self._children[action] = MCTreeNode(self, prob)

	def select(self):
		""" Select the best child of this node.
		Returns:
			A tuple of (action, next_node) with highest Q(s,a)+U(s,a)
		"""
		# Argmax_a(Q(s,a)+U(s,a))
		return max(self._children.items(), key=lambda act_node: act_node[1].get_value())

	def update(self, v):
		""" Update the three values
		"""
		# Each node's N(s,a) is updated when simulation is executed on this node, 
		# no need to update here. See MCTSearch.
		# N(s,a) = N(s,a) + 1
		#self._visit_cnt += 1
		# W(s,a) = W(s,a) + v
		self._total_action_val += v
		# Q(s,a) = W(s,a) / N(s,a)
		self._mean_action_val = self._total_action_val / self._visit_cnt



	def get_value(self):
		""" Implements PUCT Algorithm's formula for current node.
		"""
		#U(s,a)=c_punt * P(s,a) * sqrt(Parent's N(s,a)) / (1 + N(s,a))
		usa = s.c_puct * self._prior_prob * math.sqrt(self._parent._visit_cnt) / (1.0 + self._visit_cnt)
		#Q(s,a) + U(s,a)
		return self._mean_action_val + usa

	def is_leaf(self):
		"""Check if leaf node (i.e. no nodes below this have been expanded).
		"""
		return self._children == {}

	def is_root(self):
		return self._parent is None


class MCTSearch(object):
	""" Create a MC search tree. 
	"""

	def __init__(self, transformer, evaluator, max_playout = 1600):
		"""
		Arguments:
			transformer: The transforming function before the NN evaluation.
				Refers to the dihedral reflection and rotation in the paper.
			evaluator: A function that takes a state and returns (value, policies),
				where value is a float in range [-1,1]
				policies is a list of (action, prob)
		"""
		self._root = MCTreeNode(None, 1.0)
		self._state = state
		self._transformer = transformer
		self._evaluator = evaluator
		self._max_playout = max_playout

	def _playout(self, state, node):
		"""
		Arguments:
			state: current board state
			node: the node to start simulation
		"""
		# The current node is visited
		node._visit_cnt += 1

		# TODO: Do we need a max tree depth/size?
		if not node.is_leaf():
			# Greedily select next move.
			action, next_node = node.select()
			state.do_move(action)
			simulate(state, next_node)

		else:
			# Evaluate the state and get output from NN
			children_candidates, value = self._evaluator(self._transformer(state))
			# TODO: Remove invalid children

			# Check for end of game.
			if len(children_candidates) != 0:
				node.expand(children_candidates, value)

		node.update(value)

	def _select_best_move(self):
		pass

	def calc_move(self, state, dirichlet = False, prop_exp = True):
		"""Calculates the best move from the state to play.
			Remarks:
				"temperature" parameter of the two random dist is not implemented,
				because the value is set to either 1 or 0 in the paper, which can
				be controlled by toggling the option.
			Arguments:
				state: current state
				dirichlet: enable Dirichlet noise described in "Self-play" section
				prop_exp: select the final decision proportional to its exponential visit

		"""
		# The root of the tree is visited.
		self._root._n_visits += 1

		# Dirichlet noise is applied to the children of the roots, we will expand the 
		# root first

		if self._root.is_leaf():
			# Evaluate the state and get output from NN
			children_candidates, value = self._evaluator(self._transformer(state))

			# TODO: Remove the problematic children, the function could be external
			#children_candidates = remove_invalid(state, children_candidates)

			# Only create legal children
			self._root.expand(children_candidates, value)

		if dirichlet:
			# Get a list of random numbers from d=Dirichlet distribution
			dirichlet_rand = random_variate_dirichlet(s.d_alpha, s.d_epsilon, len(self._root._children))
			for action, eta in zip(self._root._children.keys(), dirichlet_rand):
				# Update the P(s,a) of all children of root
				self._root._children[action]._prior_prob = (1-s.d_epsilon) * self._root._children[action]._prior_prob + s.d_epsilon * eta

		# Do search loop while playout limit is not reached and time remains
		# TODO: Implement timing module
		for _ in range(self._max_playout):
			self._playout(state.copy(), self._root)

		# Select the best move according to the final search tree
		best_move = self._select_best_move()
		return best_move

		#return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]
			






