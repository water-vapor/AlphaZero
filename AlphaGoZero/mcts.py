import math
from AlphaGoZero.math_helper import random_variate_dirichlet, weighted_random_choice
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
        # Q(s,a) need not to be stored, since Q(s,a) always equal to W(s,a)/N(s,a)
        # self.mean_action_val = 0
        # P(s,a)
        self._prior_prob = prior_prob

    def expand(self, policy, value):
        """ Expand a leaf node according to the network evaluation.
            NO visit count is updated in this function, make sure it's updated externally
        Arguments:
            policy: a list of (action, prob) tuples returned by the network
            value: the value of this node returned by the network
        """
        # Check if the node is leaf
        if not self.is_leaf():
            return
        # Update W(s,a) for this parent node by formula W(s,a) = W(s,a) + v
        self.update(value)
        # Create valid children
        for action, prob in policy:
            # TODO: Is there an extra condition to check pass and suicide?
            # 		checking will be moved to MCTSearch
            if action not in self._children:
                self._children[action] = MCTreeNode(self, prob)

    def select(self):
        """ Select the best child of this node.
        Returns:
            A tuple of (action, next_node) with highest Q(s,a)+U(s,a)
        """
        # Argmax_a(Q(s,a)+U(s,a))
        return max(self._children.items(), key=lambda act_node: act_node[1].get_selection_value())

    def update(self, v):
        """ Update the three values
        """
        # Each node's N(s,a) is updated when simulation is executed on this node,
        # no need to update here. See MCTSearch.
        # N(s,a) = N(s,a) + 1
        # self._visit_cnt += 1
        # W(s,a) = W(s,a) + v
        self._total_action_val += v

    def get_selection_value(self, c_punt=s.c_puct):
        """ Implements PUCT Algorithm's formula for current node.
        """
        # U(s,a)=c_punt * P(s,a) * sqrt(Parent's N(s,a)) / (1 + N(s,a))
        usa = c_punt * self._prior_prob * math.sqrt(self._parent.visit_cnt) / (1.0 + self._visit_cnt)
        # Q(s,a) + U(s,a)
        return self.get_mean_action_value() + usa

    def get_mean_action_value(self):
        """Calculates Q(s,a)
        """
        # TODO: Should this value be inverted with color?
        # If yes, the signature should be changed to (self, color)
        return self._total_action_val / self._visit_cnt

    def visit(self):
        self._visit_cnt += 1

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None

    @property
    def visit_count(self):
        return self._visit_cnt

    @property
    def children(self):
        return self._children

    @property
    def prior_prob(self):
        return self._prior_prob


class MCTSearch(object):
    """ Create a MC search tree.
    """

    def __init__(self, transformer, evaluator, max_playout=1600):
        """
        Arguments:
            transformer: The transforming function before the NN evaluation.
                Refers to the dihedral reflection and rotation in the paper.
            evaluator: A function that takes a state and returns (policies, value),
                where value is a float in range [-1,1]
                policies is a list of (action, prob)
        """
        self._root = MCTreeNode(None, 1.0)
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
        node.visit()

        # TODO: Do we need a max tree depth/size?
        if not node.is_leaf():
            # Greedily select next move.
            action, next_node = node.select()
            #
            current_player = state.current_player
            state.do_move(action)
            # The result of the simulation is returned after the complete playout
            # Update this level of node with this value
            simres_value = self._playout(state, next_node)
            # Visit count is updated when. this node is first called with _playout
            # Therefore there is no visit count update in update()
            # Update relative value
            node.update(current_player * simres_value)
            # Return the same result to the parent
            return simres_value

        else:  # Node is a leaf
            # Evaluate the state and get output from NN
            children_candidates, value = self._evaluator(self._transformer(state))
            # Remove invalid children
            children_candidates = [(action, prob) for action, prob in children_candidates if state.is_legal(action)]
            # If not the end of game, expand node and terminate playout.
            # Else just terminate playout.
            if len(children_candidates) != 0:
                # Value stored (total action value) is always relative to itself
                # i.e. 1 if it wins and -1 if it loses
                # value returned by NN has -1 when white wins, multiplication will inverse
                node.expand(children_candidates, state.current_player * value)
            # Q(s,a)=W(s,a)
            # Return the black win value to update (recursively)
            else:
                return value

    def _select_best_move(self, prop_exp=True):
        """ Select the move to play according to N(s,a).
            Arguments:
                prop_exp: If enabled, select node randomly with probability
                    N(s,a)/ParentN(s,a)
        """
        if prop_exp:
            # A list of (action, selection_weight), weight is not necessarily normalized
            move_prob = [(action, node.visit_count) for action, node in self._root.children.items()]
            return weighted_random_choice(move_prob)
        else:
            # Directly select the node with most visits
            return max(self._root.children.items(), key=lambda act_node: act_node[1].visit_count)[0]

    def calc_move(self, state, dirichlet=False, prop_exp=True, d_alpha=s.d_alpha, d_epsilon=s.d_epsilon):
        """Calculates the best move from the state to play.
            Remarks:
                "temperature" parameter of the two random dist is not implemented,
                because the value is set to either 1 or 0 in the paper, which can
                be controlled by toggling the option.
            Arguments:
                state: current state
                dirichlet: enable Dirichlet noise described in "Self-play" section
                prop_exp: select the final decision proportional to its exponential visit
                d_alpha: value of alpha parameter
                d_epsilon: value of epsilon parameter

        """
        # The root of the tree is visited.
        self._root.visit()

        # Dirichlet noise is applied to the children of the roots, we will expand the
        # root first

        if self._root.is_leaf():
            # Evaluate the state and get output from NN
            children_candidates, value = self._evaluator(self._transformer(state))

            # Remove invalid children
            children_candidates = [(action, prob) for action, prob in children_candidates if state.is_legal(action)]

            # Only create legal children
            self._root.expand(children_candidates, value)

        if dirichlet:
            # Get a list of random numbers from d=Dirichlet distribution
            dirichlet_rand = random_variate_dirichlet(d_alpha, len(self._root.children))
            for action, eta in zip(self._root.children.keys(), dirichlet_rand):
                # Update the P(s,a) of all children of root
                self._root.children[action]._prior_prob = (1 - d_epsilon) * self._root.children[
                    action].prior_prob + d_epsilon * eta

        # Do search loop while playout limit is not reached and time remains
        # TODO: Implement timing module
        for _ in range(self._max_playout):
            self._playout(state.copy(), self._root)

        # Select the best move according to the final search tree
        best_move = self._select_best_move(prop_exp)
        return best_move

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
        that calc_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root._parent = None
        else:
            self._root = MCTreeNode(None, 1.0)
