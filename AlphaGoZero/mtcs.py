import numpy as np
import math
import AlphaGoZero.settings as s
from AlphaGoZero import go

class MCTreeNode(object):
	"""Tree Node in MCTS.
	"""

	def __init__(self, parent, action, prior_prob):

		self._action = action
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

	def expand(self, nn_output):
		"""
		Arguments:
			nn_output: a list of (action, prob) from policy network's evaluation of 
				the node to expand
		"""
		for action, prob in nn_output:
            if action not in self._children:
                self._children[action] = TreeNode(self, action, prob)

    def select(self):
    	"""
    	Returns:
    		A tuple of (action, next_node) with highest Q(s,a)+U(s,a)
    	"""
    	# Argmax_a(Q(s,a)+U(s,a))
    	return max(self._children.iteritems(), key=lambda act_node: act_node[1].get_value())

    def update(self, v):
    	""" Update the three values
    	"""
    	# N(s,a) = N(s,a) + 1
    	self._visit_cnt += 1
    	# W(s,a) = W(s,a) + v
    	self._total_action_val += v
    	# Q(s,a) = W(s,a) / N(s,a)
    	self._mean_action_val = self._total_action_val / self._visit_cnt


    def get_value(self):
    	""" Implements PUCT Algorithm's formula
    	"""
    	usa = s.c_puct * self._prior_prob * math.sqrt(self._parent._visit_cnt) / (1.0 + self._visit_cnt)
    	return self._mean_action_val + usa

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTSearch(object):

	def __init__(self, nn_evaluator, max_playout = 1600):
		"""
		Arguments:
			nn_evaluator: a function that takes a state and returns (value, policies),
				where value is a float in range [-1,1]
				policies is a list of (action, prob)
		"""
		self._root = MCTreeNode(None, go.PASS_MOVE, 1.0)
		self._nn_evaluator = nn_evaluator
		self._max_playout = max_playout

	def simulate(self, state, node):
		"""
		Arguments:
			state: current board state
			node: the node to start simulation
		"""
        if not node.is_leaf():
            # Greedily select next move.
            action, next_node = node.select()
            state.do_move(action)
            simulate(state, next_node)

    	else:
    		v, action_probs = self._nn_evaluator(state)
    		# Check for end of game.
	        if len(action_probs) != 0:
	            node.expand(action_probs)
	            action, node = node.select()
	            state.do_move(action) 

        # Expand leaf
        action_probs = self._policy(state)
         # Check for end of game.
        if len(action_probs) != 0:
            node.expand(action_probs)
            action, node = node.select()
            state.do_move(action) 





