import h5py as h5
import numpy as np
import os
from AlphaGoZero.Network.main import Network
from AlphaGoZero.Network.supervised import shuffled_hdf5_batch_generator
from AlphaGoZero.preprocessing.game_converter import run_game_converter
from AlphaGoZero.reinforcement_learning.sequential.util import get_current_time, combined_train_data_generator


def optimize(training_selfplays, model_to_optimize, base_dir='data', num_gpu=1, num_step=1000, num_batch=32):
    """ Update neural network models with recently generated self-play data, assuming train.h5 exists in these selfplay folders.
		Returns the name of the mewly created model (checkpoint).
		
        Arguments:
            training_selfplays: a list of model names, who has selfplay information in h5 format in the corresponding folder
            model_to_optimize: name of the model to optimize
            num_gpu: number of gpu
            num_step: number of steps in optimization
            num_batch: batch size per worker
    """
	# Load training data from h5 files
	training_h5_paths = [os.path.join(base_dir, 'selfplay', model_name, 'train.h5') for model_name in training_selfplays]
	h5_files = [h5.File(h5_path) for h5_path in training_h5_paths]
	
	# Load the model to optimize
    model = Network(num_gpu)
	model_path = os.path.join(base_dir, 'models', model_to_optimize)
    model.load(model_path)
	
    # TODO: learning rate annealing should be implemented at the network module
	# The generator takes a list of h5 file data instead of one
    train_data_generator = combined_train_data_generator(h5_files, num_batch)
	
    for step, batch in enumerate(train_data_generator):
        model.update(batch)
        if step % 100 == 0:
            print(step, "/", num_step, " done.")
        if step + 1 == num_step:
            print("checkpoint")
            break
	
	# Save to the new model
	new_model_name = get_current_time()
    model.save(os.path.join(base_dir, 'models', new_model_name)))
    del train_data_generator
	return new_model_name
