import os
from AlphaGoZero.preprocessing.game_converter import GameConverter, SizeMismatchError, NoResultError

def selfplay_to_h5(model_name, base_dir = 'data'):
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
	for (dirpath, dirname, files) in os.walk(root):
		for filename in files:
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

def sgf_to_h5(path_to_sgf, destination_path, filename):
    h5path = os.path.join(destination_path, filename)
    args = ['--outfile', h5path, '--directory', path_to_sgf]
    run_game_converter(args)
    return h5path
	
def combined_train_data_generator(h5_files, num_batch):
	pass