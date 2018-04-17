import yaml
import AlphaZero.network.main as network
import AlphaZero.interface.gtp_wrapper as gtp_wrapper
import AlphaZero.processing.state_converter as _preproc
import AlphaZero.mcts as MCTS
import numpy as np

with open('AlphaZero/config/go.yaml') as c:
    game_config = yaml.load(c)
with open('AlphaZero/config/gtp.yaml') as f:
    ext_config = yaml.load(f)
_state_tensor_converter = _preproc.StateTensorConverter(game_config)
_tensor_action_converter = _preproc.TensorActionConverter(game_config)

net = network.Network(game_config, config_file='AlphaZero/config/gtp.yaml', pretrained=ext_config['pretrained'],
                      mode='NCHW')


def nn_eval(state):
    state_np = _state_tensor_converter.state_to_tensor(state)
    result_np = net.response(np.expand_dims(state_np, 0))
    return _tensor_action_converter.tensor_to_action(result_np[0][0]), result_np[1][0]


def get_move(state):
    mcts = MCTS.MCTSearch(nn_eval, game_config, max_playout=ext_config['max_playout'])
    move, _ = mcts.calc_move_with_probs(state)
    return move


gtp_wrapper.run_gtp(get_move)
