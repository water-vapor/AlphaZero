import yaml
import os
import importlib
from tabulate import tabulate
from AlphaZero.player.mcts_player import Player as MCTSPlayer
from AlphaZero.player.cmd_player import Player as UserPlayer
from AlphaZero.player.nn_player import Player as NNPlayer
from AlphaZero.evaluator.nn_eval_seq import NNEvaluator
from AlphaZero.evaluator.dummy_eval import DummyEvaluator

with open('AlphaZero/config/play_cmd.yaml') as f:
    config = yaml.load(f)
with open('AlphaZero/config/game.yaml') as f:
    game_selection = yaml.load(f)
config_path = os.path.join('AlphaZero', 'config', game_selection['game'] + '.yaml')
if not os.path.exists(config_path):
    raise NotImplementedError("{} game config file does not exist.".format(game_selection['game']))
with open(config_path) as f:
    game_config = yaml.load(f)

board_display = {1: 'X', -1: 'O', 0: ' '}
player_type = {1: 'Human Player', 2: 'Raw NN Player', 3: 'Raw MCTS Player', 4: 'MCTS-NN Player'}


def get_player_by_type(type_id):
    if type_id == 1:
        return UserPlayer()
    elif type_id == 2:
        nn_eval = NNEvaluator()
        nn_eval.load(config['save_dir'])
        return NNPlayer(nn_eval, game_config)
    elif type_id == 3:
        return MCTSPlayer(DummyEvaluator(), game_config, config['raw_mcts'])
    elif type_id == 4:
        nn_eval = NNEvaluator()
        nn_eval.load(config['save_dir'])
        return MCTSPlayer(nn_eval, game_config, config['mcts_nn'])
    else:
        assert False


def board_display_line(line):
    return [board_display[elem] for elem in line]


def print_board(board):
    to_print = [[idx] + board_display_line(l) for idx, l in enumerate(board)]
    print(tabulate(to_print, headers=list(range(len(to_print)))))


if __name__ == '__main__':
    _game_env = importlib.import_module(game_config['env_path'])
    current_state = _game_env.GameState()
    winner = None
    for i in range(1,5):
        print(i, '. ', player_type[i])
    first_choice, second_choice = 0, 0
    while not (first_choice in [1, 2, 3, 4] and second_choice in [1, 2, 3, 4]):
        first_choice, second_choice = [int(i) for i in
                                       input('Please select the first and the second player\'s type as X Y:').split()]
    player_1 = get_player_by_type(first_choice)
    player_2 = get_player_by_type(second_choice)
    current_player = player_1
    first_players_turn = True
    while not current_state.is_end_of_game:

        print_board(current_state.board)
        current_move, _ = current_player.think(current_state)
        print(player_type[first_choice if first_players_turn else second_choice], ' plays at ', current_move)
        first_players_turn = not first_players_turn
        current_state.do_move(current_move)
        player_1.ack(current_move)
        player_2.ack(current_move)

        # change player
        if current_player == player_1:
            current_player = player_2
        else:
            current_player = player_1

    winner = current_state.get_winner()
    print_board(current_state.board)
    if winner == 0:
        print('Draw.')
    elif winner == 1:
        print(player_type[first_choice], ' wins.')
    else:
        print(player_type[second_choice], ' wins.')
