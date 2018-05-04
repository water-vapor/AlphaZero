import os
import yaml
import AlphaZero.network.main as network
import AlphaZero.interface.gtp_wrapper as gtp_wrapper
import AlphaZero.processing.state_converter as _preproc
import AlphaZero.search.mcts as MCTS
import numpy as np
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser(description='GTP player of AlphaZero.')
parser.add_argument('-m', type=str, help='Dir of model to be load. This will override the dir in config file.', default=None)
parser.add_argument('-p', type=str, help='Port for distributed tensorflow.', default=None)
parser.add_argument('-n', type=int, help='Max playout.', default=None)
args = parser.parse_args()

with open(os.path.join(os.path.dirname(__file__), 'config', 'go.yaml')) as c:
    game_config = yaml.load(c)
with open(os.path.join(os.path.dirname(__file__), 'config', 'gtp.yaml')) as f:
    ext_config = yaml.load(f)
_state_tensor_converter = _preproc.StateTensorConverter(game_config)
_tensor_action_converter = _preproc.TensorActionConverter(game_config)

port = ext_config['port'] if args.p is None else args.p
pretrained = ext_config['pretrained'] if args.m is None else False
playout = ext_config['max_playout'] if args.n is None else args.n

cluster = tf.train.ClusterSpec({'main': ['localhost:'+str(port)]})
net = network.Network(game_config, config_file='AlphaZero/config/gtp.yaml', pretrained=pretrained,
                      mode='NCHW', cluster=cluster)
if args.m is not None:
    net.load(tf.train.latest_checkpoint(args.m))


def nn_eval(state):
    state_np = _state_tensor_converter.state_to_tensor(state)
    result_np = net.response(np.expand_dims(state_np, 0))
    return _tensor_action_converter.tensor_to_action(result_np[0][0]), result_np[1][0]


def get_move(state):
    mcts = MCTS.MCTSearch(nn_eval, game_config, max_playout=playout)
    move, probs = mcts.calc_move_with_probs(state)
    return move


gtp_wrapper.run_gtp(get_move)
