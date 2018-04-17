import unittest
import yaml
import numpy as np
from AlphaZero.env.go import GameState
from AlphaZero.processing.state_converter import TensorActionConverter, StateTensorConverter, ReverseTransformer

with open('tests/go_test.yaml') as f:
    config = yaml.load(f)


def simple_board():
    """

    """
    gs = GameState(size=7)

    # make a tiny board for the sake of testing and hand-coding expected results
    #
    #         X
    #   0 1 2 3 4 5 6
    #   B W . . . . . 0
    #   B W . . . . . 1
    #   B . . . B . . 2
    # Y . . . B k B . 3
    #   . . . W B W . 4
    #   . . . . W . . 5
    #   . . . . . . . 6
    #
    # where k is a ko position (white was just captured)

    # ladder-looking thing in the top-left
    gs.do_move((0, 0))  # B
    gs.do_move((1, 0))  # W
    gs.do_move((0, 1))  # B
    gs.do_move((1, 1))  # W
    gs.do_move((0, 2))  # B

    # ko position in the middle
    gs.do_move((3, 4))  # W
    gs.do_move((3, 3))  # B
    gs.do_move((4, 5))  # W
    gs.do_move((4, 2))  # B
    gs.do_move((5, 4))  # W
    gs.do_move((5, 3))  # B
    gs.do_move((4, 3))  # W - the ko position
    gs.do_move((4, 4))  # B - does the capture

    return gs


class TestTransformers(unittest.TestCase):
    def test_forward_transformer_id_0(self):
        gs = simple_board()
        gs.transform(0)
        target = np.asarray([
            [1, 1, 1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0],
            [0, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        self.assertTrue(np.array_equal(gs.board, target))

    def test_forward_transformer_id_1_and_history(self):
        gs = simple_board()
        gs.transform(1)
        target = np.asarray([
            [1, 1, 1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0],
            [0, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        target_history_last = np.asarray([
            [1, 1, 1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0],
            [0, 0, 1, -1, 0, -1, 0],
            [0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        self.assertTrue(np.array_equal(gs.board, np.rot90(target, 1)))
        self.assertTrue(np.array_equal(gs.board_history[-1], np.rot90(target_history_last, 1)))

    def test_fwd_and_bwd_transformer(self):
        _reverse_transformer = ReverseTransformer(config).reverse_transform
        original_gs = simple_board()
        for transform_id in range(8):
            gs = simple_board()
            gs.transform(transform_id)
            fake_action_prob = [((i, j), gs.board[i][j]) for i in range(7) for j in range(7)]
            fake_action_prob.append((None, 0))
            _reverse_transformer(fake_action_prob, transform_id)
            for i in range(7 * 7):
                (x, y), p = fake_action_prob[i]
                self.assertEqual(p, original_gs.board[x][y])
