import unittest
import numpy as np
from operator import itemgetter
from AlphaGoZero.go import GameState
from AlphaGoZero.mcts import MCTreeNode, MCTSearch
from AlphaGoZero.math_helper import random_state_transform


class TestTreeNode(unittest.TestCase):
    def setUp(self):
        self.gs = GameState()
        self.node = MCTreeNode(None, 1.0)

    def test_selection(self):
        self.node.expand(random_policy(self.gs))
        action, next_node = self.node.select()
        self.assertEqual(action, (18, 18))  # according to the dummy policy below
        self.assertIsNotNone(next_node)

    def test_expansion(self):
        self.assertEqual(0, len(self.node._children))
        self.node.expand(random_policy(self.gs))
        self.assertEqual(19 * 19, len(self.node._children))
        for a, p in random_policy(self.gs):
            self.assertEqual(p, self.node._children[a]._P)


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.gs = GameState()
        self.mcts = MCTSearch(random_state_transform, policy_value_generator, max_playout=2)

    def _count_expansions(self):
        """Helper function to count the number of expansions past the root using the dummy policy
        """
        node = self.mcts._root
        expansions = 0
        # Loop over actions in decreasing probability.
        for action, _ in sorted(random_policy(self.gs), key=itemgetter(1), reverse=True):
            if action in node._children:
                expansions += 1
                node = node._children[action]
            else:
                break
        return expansions

    def test_playout(self):
        self.mcts._playout(self.gs.copy(), 8)
        # Assert that the most likely child was visited (according to the dummy policy below).
        self.assertEqual(1, self.mcts._root._children[(18, 18)]._n_visits)
        # Assert that the search depth expanded nodes 8 times.
        self.assertEqual(8, self._count_expansions())

    def test_playout_with_pass(self):
        # Test that playout handles the end of the game (i.e. passing/no moves). Mock this by
        # creating a policy that returns nothing after 4 moves.
        def stop_early_policy(state):
            if len(state.get_history()) <= 4:
                return random_policy(state)
            else:
                return []

        self.mcts = MCTSearch(random_state_transform, policy_value_generator, max_playout=2)
        self.mcts._playout(self.gs.copy(), 8)
        # Assert that (18, 18) and (18, 17) are still only visited once.
        self.assertEqual(1, self.mcts._root._children[(18, 18)]._n_visits)
        # Assert that no expansions happened after reaching the "end" in 4 moves.
        self.assertEqual(5, self._count_expansions())

    def test_get_move(self):
        move = self.mcts.calc_move(self.gs)
        self.mcts.update_with_move(move)
        # success if no errors

    def test_update_with_move(self):
        move = self.mcts.calc_move(self.gs)
        self.gs.do_move(move)
        self.mcts.update_with_move(move)
        # Assert that the new root still has children.
        self.assertTrue(len(self.mcts._root._children) > 0)
        # Assert that the new root has no parent (the rest of the tree will be garbage collected).
        self.assertIsNone(self.mcts._root._parent)
        # Assert that the next best move according to the root is (18, 17), according to the
        # dummy policy below.
        self.assertEqual((18, 17), self.mcts._root.select()[0])


# A distribution over positions that is smallest at (0,0) and largest at (18,18)
dummy_distribution = np.arange(361, dtype=np.float)
dummy_distribution = dummy_distribution / dummy_distribution.sum()


def legal_policy(state):
    moves = state.get_legal_moves(include_eyes=False)
    return zip(moves, dummy_distribution)


def random_policy(state):
    # it is MCTS's responsibility to remove the illegal children
    moves = [(x, y) for x in range(19) for y in range(19)]
    return zip(moves, dummy_distribution)


def zero_value(state):
    # it's not very confident
    return 0.0


def constant_value(state):
    return 0.5


def policy_value_generator(state):
    return random_policy(state), zero_value(state)


if __name__ == '__main__':
    unittest.main()
