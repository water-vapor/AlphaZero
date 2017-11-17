import unittest
import os
from AlphaGoZero.util import sgf_to_gamestate
from AlphaGoZero.preprocessing.game_converter import GameConverter
from AlphaGoZero.preprocessing.game_converter import run_game_converter


class TestGameConverter(unittest.TestCase):
    def setUp(self):
        features = ["board_history", "color"]
        self.converter = GameConverter(features)
        test_data_path = 'tests/test_data/sgf'
        self.test_data = [os.path.join(test_data_path, n) for n in os.listdir(test_data_path)]
        test_data_handicap_path = 'tests/test_data/sgf_with_handicap'
        self.test_data_handicap = []
        for n in os.listdir(test_data_handicap_path):
            self.test_data_handicap.append(os.path.join(test_data_handicap_path, n))

    def test_sgf_loading_normal(self):
        try:
            for test_file in self.test_data:
                with open(test_file, 'r') as f:
                    sgf_to_gamestate(f.read())
        except:
            self.fail('test_sgf_loading_normal() failed.')

    def test_sgf_loading_handicap(self):
        try:
            for test_file in self.test_data_handicap:
                with open(test_file, 'r') as f:
                    sgf_to_gamestate(f.read())
        except:
            self.fail('test_sgf_loading_handicap() failed.')

    def test_sgf_to_hdf5(self):
        try:
            self.converter.sgfs_to_hdf5(self.test_data, 'test_normal.h5', 19)
            self.converter.sgfs_to_hdf5(self.test_data_handicap, 'test_handicap.h5', 19)
            os.remove('test_normal.h5')
            os.remove('test_handicap.h5')
        except:
            self.fail('test_sgf_to_hdf5() failed.')


class TestCmdlineConverter(unittest.TestCase):
    def test_directory_conversion(self):
        try:
            args = ['--features', 'board_history,color',
                    '--outfile', '.tmp.testing.h5',
                    '--directory', 'tests/test_data/sgf/']
            run_game_converter(args)
            os.remove('.tmp.testing.h5')
        except:
            self.fail('test_directory_conversion() failed.')

    def test_directory_walk(self):
        try:
            args = ['--features', 'board_history,color',
                    '--outfile', '.tmp.testing.h5',
                    '--directory', 'tests/test_data', '--recurse']
            run_game_converter(args)
            os.remove('.tmp.testing.h5')
        except:
            self.fail('test_directory_walk() failed.')


if __name__ == '__main__':
    unittest.main()
