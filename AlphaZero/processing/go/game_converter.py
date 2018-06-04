#!/usr/bin/env python
import os
import pickle
import warnings

import h5py as h5
import numpy as np
import sgf
import yaml

import AlphaZero.env.go as go
from AlphaZero.processing.state_converter import StateTensorConverter
from AlphaZero.util import sgf_iter_states

config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'go.yaml')
with open(config_path) as c:
    config = yaml.load(c)


class SizeMismatchError(Exception):
    pass


class NoResultError(Exception):
    pass


class SearchProbsMismatchError(Exception):
    pass


class GameConverter:
    """
    Convert SGF files to network input feature files.
    """
    def __init__(self, features):
        self.feature_processor = StateTensorConverter(config, features)
        self.n_features = self.feature_processor.output_dim

    def convert_game(self, file_name, bd_size):
        """Read the given SGF file into an iterable of (input,output) pairs
        for neural network training

        Each input is a GameState converted into one-hot neural net features
        Each output is an action as an (x,y) pair (passes are skipped)

        If this game's size does not match bd_size, a SizeMismatchError is raised

        Args:
            file_name: file name
            bd_size: board size

        Returns:
            tuple: neural network input, move and result
        """

        with open(file_name, 'r') as file_object:
            state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)

        for (state, move, player, result) in state_action_iterator:
            if result is None:
                raise NoResultError()
            if state.size != bd_size:
                raise SizeMismatchError()

            # Generate features even for pass moves, since the history is different
            # Pass move is represented as (19, 0)
            nn_input = self.feature_processor.state_to_tensor(state)
            if move == go.PASS_MOVE:
                move = (19, 0)
            yield (nn_input, move, result)

    def sgfs_to_hdf5(self, sgf_files, hdf5_file, bd_size=19, ignore_errors=True, verbose=False):
        """Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file.

        The resulting file has the following properties:

            states  : dataset with shape (n_data, n_features, board width, board height)

            actions : dataset with shape (n_data, 2) (actions are stored as x,y tuples of where the move was played)

            results : dataset with shape (n_data, 1), +1 if current player wins, -1 otherwise

            file_offsets : group mapping from filenames to tuples of (index, length)

        For example, to find what positions in the dataset come from 'test.sgf':

            index, length = file_offsets['test.sgf']

            test_states = states[index:index+length]

            test_actions = actions[index:index+length]

        Args:
            sgf_files: an iterable of relative or absolute paths to SGF files
            hdf5_file: the name of the HDF5 where features will be saved
            bd_size: side length of board of games that are loaded
            ignore_errors: if True, issues a Warning when there is an unknown
            exception rather than halting. Note that sgf.ParseException and
            go.IllegalMove exceptions are always skipped
            verbose: display setting

        Returns:
            None
        """


        # make a hidden temporary file in case of a crash.
        # on success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5f = h5.File(tmp_file, 'w')

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            states = h5f.require_dataset(
                'states',
                dtype=np.uint8,
                shape=(1, self.n_features, bd_size, bd_size),
                maxshape=(None, self.n_features, bd_size, bd_size),  # 'None' == arbitrary size
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, self.n_features, bd_size, bd_size),  # approximately 1MB chunks
                compression="lzf")
            actions = h5f.require_dataset(
                'actions',
                dtype=np.uint8,
                shape=(1, 2),
                maxshape=(None, 2),
                exact=False,
                chunks=(1024, 2),
                compression="lzf")
            results = h5f.require_dataset(
                'results',
                dtype=np.int8,
                shape=(1, 1),
                maxshape=(None, 1),
                exact=False,
                chunks=(1024, 1),
                compression="lzf")

            # 'file_offsets' is an HDF5 group so that 'file_name in file_offsets' is fast
            file_offsets = h5f.require_group('file_offsets')

            # Store comma-separated list of feature planes in the scalar field 'features'. The
            # string can be retrieved using h5py's scalar indexing: h5f['features'][()]
            h5f['features'] = np.string_(','.join(self.feature_processor.feature_list))

            if verbose:
                print("created HDF5 dataset in {}".format(tmp_file))

            next_idx = 0
            for file_name in sgf_files:
                if verbose:
                    print(file_name)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                file_start_idx = next_idx
                try:
                    for state, move, result in self.convert_game(file_name, bd_size):
                        if next_idx >= len(states):
                            states.resize((next_idx + 1, self.n_features, bd_size, bd_size))
                            actions.resize((next_idx + 1, 2))
                            results.resize((next_idx + 1, 1))
                        states[next_idx] = state
                        actions[next_idx] = move
                        results[next_idx] = result
                        n_pairs += 1
                        next_idx += 1
                except go.IllegalMove:
                    warnings.warn("Illegal Move encountered in %s\n"
                                  "\tdropping the remainder of the game" % file_name)
                except sgf.ParseException:
                    warnings.warn("Could not parse %s\n\tdropping game" % file_name)
                except SizeMismatchError:
                    warnings.warn("Skipping %s; wrong board size" % file_name)
                except NoResultError:
                    warnings.warn("Skipping %s; no result or non-standard result" % file_name)
                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e),
                                      stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        # '/' has special meaning in HDF5 key names, so they
                        # are replaced with ':' here
                        file_name_key = file_name.replace('/', ':')
                        file_offsets[file_name_key] = [file_start_idx, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("sgfs_to_hdf5 failed")
            os.remove(tmp_file)
            raise e

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
        os.rename(tmp_file, hdf5_file)

    def selfplay_to_hdf5(self, sgf_pkl_files, hdf5_file, bd_size=19, ignore_errors=True, verbose=False):
        """Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file.

        The resulting file has the following properties:

            states  : dataset with shape (n_data, n_features, board width, board height)

            actions : dataset with shape (n_data, 2) (actions are stored as x,y tuples of where the move was played)

            results : dataset with shape (n_data, 1), +1 if current player wins, -1 otherwise

            file_offsets : group mapping from filenames to tuples of (index, length)

        For example, to find what positions in the dataset come from 'test.sgf':

            index, length = file_offsets['test.sgf']

            test_states = states[index:index+length]

            test_actions = actions[index:index+length]

        Args:
            sgf_pkl_files: an iterable of relative or absolute paths to SGF and PKL files
            hdf5_file: the name of the HDF5 where features will be saved
            bd_size: side length of board of games that are loaded
            ignore_errors: if True, issues a Warning when there is an unknown
            exception rather than halting. Note that sgf.ParseException and
            go.IllegalMove exceptions are always skipped
            verbose: display setting

        Returns:
            None
        """

        # make a hidden temporary file in case of a crash.
        # on success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5f = h5.File(tmp_file, 'w')

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            states = h5f.require_dataset(
                'states',
                dtype=np.uint8,
                shape=(1, self.n_features, bd_size, bd_size),
                maxshape=(None, self.n_features, bd_size, bd_size),  # 'None' == arbitrary size
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, self.n_features, bd_size, bd_size),  # approximately 1MB chunks
                compression="lzf")
            search_probs = h5f.require_dataset(
                'search_probs',
                dtype=np.float32,
                shape=(1, bd_size * bd_size + 1),
                maxshape=(None, bd_size * bd_size + 1),
                exact=False,
                chunks=(1024, bd_size * bd_size + 1),
                compression="lzf")
            results = h5f.require_dataset(
                'results',
                dtype=np.int8,
                shape=(1, 1),
                maxshape=(None, 1),
                exact=False,
                chunks=(1024, 1),
                compression="lzf")

            # 'file_offsets' is an HDF5 group so that 'file_name in file_offsets' is fast
            file_offsets = h5f.require_group('file_offsets')

            # Store comma-separated list of feature planes in the scalar field 'features'. The
            # string can be retrieved using h5py's scalar indexing: h5f['features'][()]
            h5f['features'] = np.string_(','.join(self.feature_processor.feature_list))

            if verbose:
                print("created HDF5 dataset in {}".format(tmp_file))

            next_idx = 0
            pkl_idx = 0
            for file_name, pkl_name in sgf_pkl_files:
                if verbose:
                    print(file_name, pkl_name)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                file_start_idx = next_idx
                try:
                    for state, move, result in self.convert_game(file_name, bd_size):
                        if next_idx >= len(states):
                            states.resize((next_idx + 1, self.n_features, bd_size, bd_size))
                            # actions.resize((next_idx + 1, 2))
                            results.resize((next_idx + 1, 1))
                        states[next_idx] = state
                        # actions[next_idx] = move
                        results[next_idx] = result
                        n_pairs += 1
                        next_idx += 1
                    with open(pkl_name, 'rb') as f:
                        search_probs_list = pickle.load(f)
                    for search_prob in search_probs_list:
                        search_probs.resize((pkl_idx + 1, bd_size * bd_size + 1))
                        prob_vector = np.zeros(bd_size * bd_size + 1)
                        for action, prob in search_prob:
                            if action is go.PASS_MOVE:
                                prob_idx = bd_size * bd_size
                            else:
                                move_x, move_y = action
                                prob_idx = bd_size * move_x + move_y
                            prob_vector[prob_idx] = prob
                        search_probs[pkl_idx] = prob_vector
                        pkl_idx += 1
                    if next_idx != pkl_idx:
                        raise SearchProbsMismatchError()

                except go.IllegalMove:
                    warnings.warn("Illegal Move encountered in %s\n"
                                  "\tdropping the remainder of the game" % file_name)
                except sgf.ParseException:
                    warnings.warn("Could not parse %s\n\tdropping game" % file_name)
                except SizeMismatchError:
                    warnings.warn("Skipping %s; wrong board size" % file_name)
                except NoResultError:
                    warnings.warn("Skipping %s; no result or non-standard result" % file_name)
                except SearchProbsMismatchError:
                    warnings.warn(
                        "Skipping %s; move numbers in sgf game record is inconsistant with that in search probs"
                        % file_name)
                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e),
                                      stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        # '/' has special meaning in HDF5 key names, so they
                        # are replaced with ':' here
                        file_name_key = file_name.replace('/', ':')
                        file_offsets[file_name_key] = [file_start_idx, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("sgfs_to_hdf5 failed")
            os.remove(tmp_file)
            raise e

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
        os.rename(tmp_file, hdf5_file)


def run_game_converter(cmd_line_args=None):
    """Run conversions.

    Args:
        cmd_line_args: command-line args may be passed in as a list

    Returns:
        None
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the neural network model.',
        epilog="Available features are: board_history and color")
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)  # noqa: E501
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for SGF files",
                        default=False, action="store_true")  # noqa: E501
    parser.add_argument("--directory", "-d",
                        help="Directory containing SGF files to process. if not present, expects files from stdin",
                        default=None)  # noqa: E501
    parser.add_argument("--size", "-s",
                        help="Size of the game board. SGFs not matching this are discarded with a warning", type=int,
                        default=19)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False,
                        action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    feature_list = ["board_history", "color"]

    if args.verbose:
        print("using features", feature_list)

    converter = GameConverter(feature_list)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _walk_all_sgfs(root):
        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for (dirpath, dirname, _files) in os.walk(root):
            for filename in _files:
                if _is_sgf(filename):
                    # yield the full (relative) path to the file
                    yield os.path.join(dirpath, filename)

    def _list_sgfs(path):
        """helper function to get all SGF files in a directory (does not recurse)
        """
        _files = os.listdir(path)
        return (os.path.join(path, f) for f in _files if _is_sgf(f))

    # get an iterator of SGF files according to command line args
    if args.directory:
        if args.recurse:
            files = _walk_all_sgfs(args.directory)
        else:
            files = _list_sgfs(args.directory)
    else:
        files = (f.strip() for f in sys.stdin if _is_sgf(f))

    converter.sgfs_to_hdf5(files, args.outfile, bd_size=args.size, verbose=args.verbose)


if __name__ == '__main__':
    run_game_converter()
