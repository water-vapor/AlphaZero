import os, re
import numpy as np
from AlphaZero.processing.go.game_converter import GameConverter


def selfplay_to_h5(model_name, base_dir='data'):
    """ Takes a model that has just generated the selfplay data, combine everything into a single h5 file.
        And store the h5 file as 'train.h5' in the same folder.

        Arguments:
            model_name: name of the model
            base_dir: the directory containing the folder selfplay.
    """
    feature_list = ["board_history", "color"]
    converter = GameConverter(feature_list)

    # From game converter
    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _walk_all_sgfs(root):

        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for (dirpath, dirname, _files) in os.walk(root):
            for filename in _files:
                if _is_sgf(filename):
                    # find the corresponding pkl
                    pkl_name = filename.strip()[:-4] + '.pkl'
                    if os.path.exists(os.path.join(dirpath, pkl_name)):
                        # yield the full (relative) path to the file
                        yield os.path.join(dirpath, filename), os.path.join(dirpath, pkl_name)

    files = _walk_all_sgfs(os.path.join(base_dir, 'selfplay', model_name))
    converter.selfplay_to_hdf5(files, os.path.join(base_dir, 'selfplay', model_name, 'train.h5'), 19)


def get_current_time():
    return '_'.join(re.findall('\d+', str(np.datetime64('now'))))


def combined_selfplay_h5_train_data_generator(h5_files, num_batch):
    state_datasets = [h5f['states'] for h5f in h5_files]
    search_probs_datasets = [h5f['search_probs'] for h5f in h5_files]
    result_datasets = [h5f['results'] for h5f in h5_files]
    state_dataset = np.concatenate(([ds.value for ds in state_datasets]), axis=0)
    search_probs_dataset = np.concatenate(([ds.value for ds in search_probs_datasets]), axis=0)
    result_datasets = np.concatenate(([ds.value for ds in result_datasets]), axis=0)
    n_total_data = state_dataset.shape[0]
    shuffle_indices = np.random.permutation(n_total_data)
    indices = shuffle_indices[0:n_total_data]
    return _selfplay_shuffled_hdf5_batch_generator(state_dataset, search_probs_dataset, result_datasets, indices,
                                                   num_batch)


def _selfplay_shuffled_hdf5_batch_generator(state_dataset, search_probs_dataset, result_dataset,
                                            indices, batch_size):
    """A generator of batches of training data for use with the fit_generator function
    of Keras. Data is accessed in the order of the given indices for shuffling.
    """
    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape)
    Ybatch = np.zeros((batch_size, game_size * game_size + 1), dtype=np.float32)
    Zbatch = np.zeros(batch_size)
    batch_idx = 0
    while True:
        for data_idx in indices:
            state = np.array([plane for plane in state_dataset[data_idx]])
            search_probs = search_probs_dataset[data_idx]
            result = result_dataset[data_idx]
            Xbatch[batch_idx] = state
            Ybatch[batch_idx] = search_probs
            Zbatch[batch_idx] = result
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch, Zbatch)


def shuffled_npy_batch_generator(state_dataset, search_probs_dataset, result_dataset, indices, batch_size):
    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape)
    Ybatch = np.zeros((batch_size, game_size * game_size + 1), dtype=np.float32)
    Zbatch = np.zeros(batch_size)
    batch_idx = 0
    while True:
        for data_idx in indices:
            state = np.array([plane for plane in state_dataset[data_idx]])
            search_probs = search_probs_dataset[data_idx]
            result = result_dataset[data_idx]
            Xbatch[batch_idx] = state
            Ybatch[batch_idx] = search_probs
            Zbatch[batch_idx] = result
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch, Zbatch)
