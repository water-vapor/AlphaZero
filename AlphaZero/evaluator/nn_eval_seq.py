import yaml
import importlib
import numpy as np
import AlphaZero.network.main as network

with open('AlphaZero/config/game.yaml') as f:
    game_selection = yaml.load(f)['game']
with open('AlphaZero/config/' + game_selection + '.yaml') as c:
    game_config = yaml.load(c)

_preproc = importlib.import_module(game_config['state_converter_path'])
_state_tensor_converter = _preproc.StateTensorConverter(game_config)
_tensor_action_converter = _preproc.TensorActionConverter(game_config)


class NNEvaluator:
    def __init__(self):
        self.net = network.Network(game_config)

    def load(self, save_dir):
        self.net.load(save_dir)

    def save(self, save_dir):
        self.net.save(save_dir)

    def eval(self, state):
        state_np = _state_tensor_converter.state_to_tensor(state)
        result_np = self.net.response(np.expand_dims(state_np, 0))
        return _tensor_action_converter.tensor_to_action(result_np[0][0]), result_np[1][0]
