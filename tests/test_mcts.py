import unittest
import numpy as np
import yaml
from operator import itemgetter
from AlphaZero.env.go import GameState
from AlphaZero.mcts import MCTreeNode, MCTSearch

with open('tests/go_test.yaml') as f:
    config = yaml.load(f)


class TestTreeNode(unittest.TestCase):
    def setUp(self):
        self.gs = GameState()
        self.node = MCTreeNode(None, 1.0)

    def test_selection(self):
        self.node.expand(random_policy(self.gs), zero_value(self.gs))
        self.node.visit()
        action, next_node = self.node.select()
        self.assertEqual(action, (18, 18))  # according to the dummy policy below
        self.assertIsNotNone(next_node)

    def test_expansion(self):
        self.assertEqual(0, len(self.node._children))
        self.node.expand(random_policy(self.gs), zero_value(self.gs))
        self.assertEqual(19 * 19 + 1, len(self.node._children))
        for a, p in random_policy(self.gs):
            self.assertEqual(p, self.node._children[a].prior_prob)


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.gs = GameState()
        self.mcts = MCTSearch(policy_value_generator(random_policy, zero_value), config, max_playout=1)

    def _tree_traversal_helper(self, node):
        if node._children != {}:
            expansions = 1
        else:
            return 0
        for action, _ in sorted(random_policy(self.gs), key=itemgetter(1), reverse=True):
            if action in node._children:
                expansions += self._tree_traversal_helper(node._children[action])
        return expansions

    def _count_expansions(self):
        """Helper function to count the number of expansions past the root using the dummy policy
        """
        node = self.mcts._root
        return self._tree_traversal_helper(node) - 1  # remove root node count

    def test_playout(self):
        for _ in range(8):
            self.mcts._playout(self.gs.copy(), self.mcts._root)
        # for i in range(19):
        #     for j in range(19):
        #         if self.mcts._root._children[(i, j)].visit_count >= 1:
        #             print((i, j), self.mcts._root._children[(i, j)].visit_count)
        self.mcts._playout(self.gs.copy(), self.mcts._root)
        # Assert that the most likely child was visited (according to the dummy policy below).
        self.assertEqual(1, self.mcts._root._children[(18, 18)].visit_count)
        # Assert that the search depth expanded nodes 8 times.
        self.assertEqual(8, self._count_expansions())

    # def test_playout_with_pass(self):
    #     # Test that playout handles the end of the game (i.e. passing/no moves). Mock this by
    #     # creating a policy that returns nothing after 4 moves.
    #     def stop_early_policy(state):
    #         if state.turns <= 4:
    #             return random_policy(state)
    #         else:
    #             return []
    #
    #     self.mcts = MCTSearch(policy_value_generator(stop_early_policy, zero_value), config, max_playout=8)
    #     for _ in range(8):
    #         self.mcts._playout(self.gs.copy(), self.mcts._root)
    #     # Assert that (18, 18) and (18, 17) are still only visited once.
    #     self.assertEqual(1, self.mcts._root._children[(18, 18)].visit_count)
    #     # Assert that no expansions happened after reaching the "end" in 4 moves.
    #     self.assertEqual(5, self._count_expansions())

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
        self.assertEqual((18, 18), move)
        self.assertEqual((18, 17), self.mcts._root.select()[0])


# A distribution over positions that is smallest at (0,0) and largest at (18,18)
dummy_distribution = np.arange(361, dtype=np.float)
dummy_distribution = dummy_distribution / dummy_distribution.sum()


def random_policy(state):
    # it is MCTS's responsibility to remove the illegal children
    moves = [(x, y) for x in range(19) for y in range(19)]
    policy = list(zip(moves, dummy_distribution))
    policy.append((None, 0))
    return policy


def zero_value(state):
    # it's not very confident
    return 0.0


def constant_value(state):
    return 0.5


def policy_value_generator(policy, value):
    def policy_value(state):
        return policy(state), value(state)

    return policy_value


if __name__ == '__main__':
    unittest.main()
