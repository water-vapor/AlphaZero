import h5py as h5
import numpy as np
import os
from AlphaGoZero.Network.main import Network
from AlphaGoZero.Network.supervised import shuffled_hdf5_batch_generator
from AlphaGoZero.preprocessing.game_converter import run_game_converter


def sgf_to_h5(path_to_sgf, destination_path, filename):
    h5path = os.path.join(destination_path, filename)
    args = ['--outfile', h5path, '--directory', path_to_sgf]
    run_game_converter(args)
    return h5path


def optimize(training_h5_file, model_path, output_path, num_gpu=1, num_step=1000, num_batch=32):
    """ Update neural network with recently generated self-play data.
        Arguments:
            training_h5_file: recently generated 500,000 self-play games in hdf5 format
            model_path: the model to optimize
            output_path: the model after optimize (checkpoint)
            num_gpu: number of gpu
            num_step: number of steps in optimization
            num_batch: batch size per worker
    """
    training_data = h5.File(training_h5_file)
    model = Network(num_gpu)
    model.load(model_path)
    # TODO: learning rate annealing should be implemented at the network module
    train_data_generator = shuffled_hdf5_batch_generator(
        training_data["states"],
        training_data["actions"],
        training_data["results"],
        np.random.permutation(len(training_data["states"])),
        num_batch)
    for step, batch in enumerate(train_data_generator):
        model.update(batch)
        if step % 100 == 0:
            print(step, "/", num_step, " done.")
        if step + 1 == num_step:
            print("checkpoint")
            break
    model.save(output_path)
    del train_data_generator
