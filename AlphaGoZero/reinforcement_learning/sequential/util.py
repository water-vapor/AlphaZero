
def selfplay_to_h5(model_name, base_dir = 'data'):
	""" Takes a model that has just generated the selfplay data, combine everything into a single h5 file.
		And store the h5 file as 'train.h5' in the same folder.
		
		Arguments:
			model_name: name of the model
			base_dir: the directory containing the folder selfplay.
	"""
	pass
	
def get_current_time():
    return '_'.join(re.findall('\d+', str(np.datetime64('now'))))

def sgf_to_h5(path_to_sgf, destination_path, filename):
    h5path = os.path.join(destination_path, filename)
    args = ['--outfile', h5path, '--directory', path_to_sgf]
    run_game_converter(args)
    return h5path