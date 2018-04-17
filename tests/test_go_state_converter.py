import unittest
import numpy as np
import yaml
from AlphaZero.env.go import GameState
from AlphaZero.processing.state_converter import StateTensorConverter

with open('AlphaZero/config/go.yaml') as f:
    config = yaml.load(f)

def empty_board():
    """
    """
    gs = GameState(size=7)
    return gs


def start_board():
    """
    """
    gs = GameState(size=7)
    gs.do_move((1, 1))
    gs.do_move((2, 2))
    return gs


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


def bdhistory_to_boards(processor, gs):
    own = [(processor.state_to_tensor(gs)[0])[2 * i] for i in range(config['history_step'])]
    opp = [(processor.state_to_tensor(gs)[0])[2 * i + 1] for i in range(config['history_step'])]
    return own, opp


class TestPreprocessingFeatures(unittest.TestCase):
    """Test the functions in processing.py

    note that the hand-coded features look backwards from what is depicted
    in simple_board() because of the x/y column/row transpose thing (i.e.
    numpy is typically thought of as indexing rows first, but we use (x,y)
    indexes, so a numpy row is like a go column and vice versa)
    """

    def test_get_board_history_mid(self):
        gs = simple_board()
        pp = StateTensorConverter(config, ["board_history"])
        # In simple board, the current player is white
        own, opp = bdhistory_to_boards(pp, gs)

        white_pos_first = np.asarray([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])
        black_pos_first = np.asarray([
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])

        white_pos_3rd = np.asarray([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])
        black_pos_3rd = np.asarray([
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])

        white_pos_last = np.asarray([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])
        black_pos_last = np.asarray([
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])

        # check history length
        self.assertEqual(len(own), config['history_step'])
        self.assertEqual(len(opp), config['history_step'])
        # check return value against hand-coded expectation
        # (given that current_player is white)
        # manually compare boards
        self.assertTrue(np.all(white_pos_first == own[0]))
        self.assertTrue(np.all(white_pos_3rd == own[2]))
        self.assertTrue(np.all(white_pos_last == own[-1]))

        self.assertTrue(np.all(black_pos_first == opp[0]))
        self.assertTrue(np.all(black_pos_3rd == opp[2]))
        self.assertTrue(np.all(black_pos_last == opp[-1]))

    def test_get_board_history_start(self):
        gs = start_board()
        pp = StateTensorConverter(config, ["board_history"])
        # In simple board, the current player is white
        own, opp = bdhistory_to_boards(pp, gs)

        padded_board = np.zeros((7, 7))

        white_pos_1stmove = np.zeros((7, 7))

        black_pos_1stmove = np.asarray([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])

        white_pos_last = np.asarray([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])

        black_pos_last = np.asarray([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])

        # check history length
        self.assertEqual(len(own), config['history_step'])
        self.assertEqual(len(opp), config['history_step'])
        # check return value against hand-coded expectation
        # (given that current_player is black)
        # manually compare boards
        for i in range(6):
            self.assertTrue(np.all(own[i] == padded_board))
            self.assertTrue(np.all(opp[i] == padded_board))

        self.assertTrue(np.all(black_pos_1stmove == own[-2]))
        self.assertTrue(np.all(black_pos_last == own[-1]))

        self.assertTrue(np.all(white_pos_1stmove == opp[-2]))
        self.assertTrue(np.all(white_pos_last == opp[-1]))

    def test_color(self):
        """

        """
        gs = simple_board()
        pp = StateTensorConverter(config, ["color"])
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))
        # white is the current player, current player is NOT black, the value should be 0
        self.assertTrue(np.all(feature == np.zeros((7, 7))))

        gs2 = start_board()
        feature2 = pp.state_to_tensor(gs2)[0].transpose((1, 2, 0))
        # current player is black
        self.assertTrue(np.all(feature2 == np.ones((7, 7))))

    def test_feature_concatenation(self):
        """

        """

        gs = simple_board()
        pp = StateTensorConverter(config, ["board_history", "color"])
        pp2 = StateTensorConverter(config, ["board_history"])
        pp3 = StateTensorConverter(config, ["color"])
        feature = pp.state_to_tensor(gs)[0]
        bd_h = pp2.state_to_tensor(gs)[0]
        bd_c = pp3.state_to_tensor(gs)[0]

        self.assertTrue(np.array_equal(feature[:-1, :, :],  bd_h))
        self.assertTrue(np.array_equal(feature[-1:, :, :], bd_c))


if __name__ == '__main__':
    unittest.main()
