import yaml
import os
import importlib
import AlphaZero.network.main as network
import AlphaZero.processing.state_converter as _preproc
from AlphaZero.game.player import Player
import numpy as np

with open('AlphaZero/config/play_cmd.yaml') as f:
    config = yaml.load(f)
config_path = os.path.join('AlphaZero', 'config', config['game'] + '.yaml')
if not os.path.exists(config_path):
    raise NotImplementedError("{} game config file does not exist.".format(config['game']))
with open(config_path) as f:
    game_config = yaml.load(f)

_state_tensor_converter = _preproc.StateTensorConverter(game_config)
_tensor_action_converter = _preproc.TensorActionConverter(game_config)

net = network.Network(game_config)
net.load(config['save_dir'])


class NN_eval:
    def eval(self, state):
        state_np = _state_tensor_converter.state_to_tensor(state)
        result_np = net.response(np.expand_dims(state_np, 0))
        return _tensor_action_converter.tensor_to_action(result_np[0][0]), result_np[1][0]


class UserPlayer:
    def __init__(self):
        pass

    def think(self, state):
        legal = False
        while not legal:
            x, y = [int(i) for i in input('Please input the coordinate as X Y: ').split()]
            legal = state.is_legal((x, y))
        return (x, y), None

    def ack(self, move):
        pass


if __name__ == '__main__':
    _game_env = importlib.import_module(game_config['env_path'])
    current_state = _game_env.GameState()
    winner = None
    nn_player = Player(NN_eval(), game_config, config)
    first_i = None
    while first_i != 'Y' and first_i != 'N':
        first_i = input('Do you want to play first? (Y/N)')
    if first_i == 'Y':
        player_1 = UserPlayer()
        player_2 = nn_player
    else:
        player_1 = nn_player
        player_2 = UserPlayer()

    current_player = player_1
    while not current_state.is_end_of_game:

        print(current_state.board)
        print('\n')
        current_move, _ = current_player.think(current_state)
        current_state.do_move(current_move)
        player_1.ack(current_move)
        player_2.ack(current_move)

        # change player
        if current_player == player_1:
            current_player = player_2
        else:
            current_player = player_1

    winner = current_state.get_winner()
    if winner == None:
        print('Draw.')
    elif winner == 1:
        if first_i == 'Y':
            print('You win!')
        else:
            print('You lose.')
    else:
        if first_i == 'N':
            print('You win!')
        else:
            print('You lose.')
