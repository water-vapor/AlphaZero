import importlib
import math

from numpy.random import randint

from AlphaZero.search.math_helper import random_variate_dirichlet, weighted_random_choice

# Parameter for PUCT Algorithm
c_punt = 5.0


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
        # But Q(s,a) should be initialized to zero.
        # self.mean_action_val = 0
        # P(s,a)
        self._prior_prob = prior_prob

    def expand(self, policy, value):
        """Expand a leaf node according to the network evaluation.
        NO visit count is updated in this function, make sure it's updated externally.

        Args:
            policy: a list of (action, prob) tuples returned by the network
            value: the value of this node returned by the network

        Returns:
            None
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
            tuple: A tuple of (action, next_node) with highest Q(s,a)+U(s,a)
        """
        # Argmax_a(Q(s,a)+U(s,a))
        return max(self._children.items(), key=lambda act_node: act_node[1].get_selection_value())

    def update(self, v):
        """ Update the three values

        Args:
            v: value

        Returns:
            None
        """

        # Each node's N(s,a) is updated when simulation is executed on this node,
        # no need to update here. See MCTSearch.
        # N(s,a) = N(s,a) + 1
        # self._visit_cnt += 1
        # W(s,a) = W(s,a) + v
        self._total_action_val += v

    def get_selection_value(self):
        """Implements PUCT Algorithm's formula for current node.

        Returns:
            None
        """

        # U(s,a)=c_punt * P(s,a) * sqrt(Parent's N(s,a)) / (1 + N(s,a))
        usa = c_punt * self._prior_prob * math.sqrt(self._parent.visit_count) / (1.0 + self._visit_cnt)
        # Q(s,a) + U(s,a)
        return self.get_mean_action_value() + usa

    def get_mean_action_value(self):
        """Calculates Q(s,a)

        Returns:
            real: mean action value
        """

        # TODO: Should this value be inverted with color?
        # If yes, the signature should be changed to (self, color)
        if self._visit_cnt == 0:
            return 0
        return self._total_action_val / self._visit_cnt

    def visit(self):
        """Increment the visit count.

        Returns:
            None
        """
        self._visit_cnt += 1

    def is_leaf(self):
        """Checks if it is a leaf node (i.e. no nodes below this have been expanded).

        Returns:
            bool: if the current node is leaf.
        """

        return self._children == {}

    def is_root(self):
        """Checks if it is a root node.

        Returns:
            bool: if the current node is root.
        """
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

    @prior_prob.setter
    def prior_prob(self, value):
        self._prior_prob = value


class MCTSearch(object):
    """ Create a Monto Carlo search tree.
    """

    def __init__(self, evaluator, game_config, max_playout=1600):
        """
        Arguments:
            evaluator: A function that takes a state and returns (policies, value),
                where value is a float in range [-1,1]
                policies is a list of (action, prob)
            game_config: Game configuration file
        """
        self._root = MCTreeNode(None, 1.0)
        self._evaluator = evaluator
        self._max_playout = max_playout
        self.d_alpha = game_config['d_alpha']
        self.d_epsilon = game_config['d_epsilon']
        self._transform_types = game_config['transform_types']
        if self._transform_types == 0 or self._transform_types == 1:
            self.enable_transform = False
        else:
            self.enable_transform = True
            self._sc = importlib.import_module(game_config['state_converter_path'])
            self._reverse_transformer = self._sc.ReverseTransformer(game_config)
            self._reverse_transform = self._reverse_transformer.reverse_transform

    def _playout(self, state, node):
        """
        Recursively executes playout from the current node.
        Args:
            state: current board state
            node: the node to start simulation

        Returns:
            real: the action value of the current node
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
            node.update(-current_player * simres_value)
            # Return the same result to the parent
            return simres_value

        else:  # Node is a leaf
            # Evaluate the state and get output from NN
            if self.enable_transform:
                # Generate a random transform ID
                random_transform_id = randint(self._transform_types)
                state_eval = state.copy()
                state_eval.transform(random_transform_id)
                transformed_children_candidates, value = self._evaluator(state_eval)
                self._reverse_transform(transformed_children_candidates, random_transform_id)
                children_candidates = transformed_children_candidates
            else:
                children_candidates, value = self._evaluator(state)
            # Remove invalid children
            children_candidates = [(action, prob) for action, prob in children_candidates if state.is_legal(action)]
            # If not the end of game, expand node and terminate playout.
            # Else just terminate playout.
            if len(children_candidates) != 0 and not state.is_end_of_game:
                # Value stored (total action value) is always relative to itself
                # i.e. 1 if it wins and -1 if it loses
                # value returned by NN has -1 when white wins, multiplication will inverse
                node.expand(children_candidates, -state.current_player * value)
            # Q(s,a)=W(s,a)
            # Return the black win value to update (recursively)
            else:
                # No valid move, game should end. Overwrite the value with the real game result.
                # Game result is absolute: 1, 0, or -1
                value = state.get_winner()
                node.update(-state.current_player * value)
            return value

    def _get_search_probs(self):
        """ Calculate the search probabilities exponentially to the visit counts.
            Returns:
                list: a list of (action, probs)
        """
        # A list of (action, selection_weight), weight is not normalized
        moves = [(action, node.visit_count) for action, node in self._root.children.items()]
        total = sum([count for _, count in moves])
        normalized_probs = [(action, count / total) for action, count in moves]
        return normalized_probs

    def _calc_move(self, state, dirichlet=False):
        """ Performs MCTS.

            "temperature" parameter of the two random dist is not implemented,
            because the value is set to either 1 or 0 in the paper, which can
            be controlled by toggling the option.

            Args:
                state: current state
                dirichlet: enable Dirichlet noise described in "Self-play" section

            Returns:
                None

        """
        # The root of the tree is visited.
        self._root.visit()

        # Dirichlet noise is applied to the children of the roots, we will expand the
        # root first

        if self._root.is_leaf():
            # Evaluate the state and get output from NN
            if self.enable_transform:
                # Generate a random transform ID
                random_transform_id = randint(self._transform_types)
                state_eval = state.copy()
                state_eval.transform(random_transform_id)
                transformed_children_candidates, value = self._evaluator(state_eval)
                self._reverse_transform(transformed_children_candidates, random_transform_id)
                children_candidates = transformed_children_candidates
            else:
                children_candidates, value = self._evaluator(state)

            # Remove invalid children
            children_candidates = [(action, prob) for action, prob in children_candidates if state.is_legal(action)]

            # Only create legal children
            self._root.expand(children_candidates, value)

        if dirichlet:
            # Get a list of random numbers from d=Dirichlet distribution
            dirichlet_rand = random_variate_dirichlet(self.d_alpha, len(self._root.children))
            for action, eta in zip(self._root.children.keys(), dirichlet_rand):
                # Update the P(s,a) of all children of root
                self._root.children[action].prior_prob = (1 - self.d_epsilon) * self._root.children[
                    action].prior_prob + self.d_epsilon * eta

        # Do search loop while playout limit is not reached and time remains
        # TODO: Implement timing module
        for _ in range(self._max_playout):
            self._playout(state.copy(), self._root)

    def calc_move(self, state, dirichlet=False, prop_exp=True):
        """ Calculates the best move

        Args:
            state: current state
            dirichlet: enable Dirichlet noise described in "Self-play" section
            prop_exp: select the final decision proportional to its exponential visit

        Returns:
            tuple: the calculated result (x, y)

        """
        self._calc_move(state, dirichlet)
        # Select the best move according to the final search tree
        # select node randomly with probability: N(s,a)/ParentN(s,a)
        if prop_exp:
            # A list of (action, selection_weight), weight is not necessarily normalized
            return weighted_random_choice(self._get_search_probs())
        else:
            # Directly select the node with most visits
            return max(self._root.children.items(), key=lambda act_node: act_node[1].visit_count)[0]

    def calc_move_with_probs(self, state, dirichlet=False):
        """ Calculates the best move, and return the search probabilities.
            This function should only be used for self-play.

        Args:
            state: current state
            dirichlet: enable Dirichlet noise described in "Self-play" section

        Returns:
            tuple: the result (x, y) and a list of (action, probs)
        """
        self._calc_move(state, dirichlet)
        probs = self._get_search_probs()
        result = weighted_random_choice(self._get_search_probs())
        return result, probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
        that calc_move() has been called already. Siblings of the new root will be garbage-collected.
        Returns:
            None
        """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root._parent = None
        else:
            self._root = MCTreeNode(None, 1.0)
