import h5py as h5
import numpy as np
import os
import numpy as np
from AlphaGoZero.Network.main import Network
from AlphaGoZero.Network.supervised import shuffled_hdf5_batch_generator
from AlphaGoZero.preprocessing.game_converter import run_game_converter
from AlphaGoZero.reinforcement_learning.sequential.util import get_current_time, shuffled_npy_batch_generator


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
	# Load training data from npy files
	training_data_paths = [os.path.join(base_dir, 'selfplay', model_name, 'train.npy') for model_name in training_selfplays]
	state_dataset = np.zeros((0,17,19,19))
	probs_dataset = np.zeros((0,362))
	result_dataset = np.zeros((0))
	for training_data_file in training_data_paths:
		with open(training_data_file) as f:
			training_data = np.load(f)
		state_np, probs_np, result_np = training_data[0], training_data[1], training_data[2]
		state_dataset = np.concatenate([state_dataset, state_np])
		probs_dataset = np.concatenate([probs_dataset, probs_np])
		result_dataset = np.concatenate([result_dataset, result_np])
	
	# Load the model to optimize
    model = Network(num_gpu)
	model_path = os.path.join(base_dir, 'models', model_to_optimize)
    model.load(model_path)
	
    # TODO: learning rate annealing should be implemented at the network module
	n_data = state_dataset.shape[0]
	shuffle_indices = np.random.permutation(n_total_data)
    indices = shuffle_indices[0:n_total_data]
    train_data_generator = shuffled_npy_batch_generator(state_dataset, probs_dataset, result_dataset, indices, num_batch)
	
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
