import importlib
import os

import numpy as np
import yaml

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'game.yaml')) as f:
    game_selection = yaml.load(f)['game']
with open(os.path.join(os.path.dirname(__file__), '..', 'config', game_selection + '.yaml')) as c:
    game_config = yaml.load(c)

_preproc = importlib.import_module(game_config['state_converter_path'])
_tensor_action_converter = _preproc.TensorActionConverter(game_config)


class DummyEvaluator:
    def __init__(self):
        pass

    def load(self, save_dir):
        pass

    def save(self, save_dir):
        pass

    def eval(self, state):
        dims = game_config['flat_move_output']
        return _tensor_action_converter.tensor_to_action(np.full((dims,), 1 / dims)), 0
