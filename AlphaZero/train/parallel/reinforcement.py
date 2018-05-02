import argparse
import multiprocessing as mp
import os
import time
import tensorflow as tf
import yaml
import AlphaZero.evaluator.nn_eval_parallel as nn_eval
import AlphaZero.train.parallel.evaluator as evaluator
import AlphaZero.train.parallel.optimization as optimization
import AlphaZero.train.parallel.selfplay as selfplay
from AlphaZero.train.parallel.util import *

if __name__ == '__main__':
    # Read the name of the game from cmd, load name.yaml from config folder
    parser = argparse.ArgumentParser(description='Performs reinforcement learning of AlphaZero.')
    # parser.add_argument("--game", '-g', help="Name of the game, in lower case.", type=str, default="go")
    args = parser.parse_args()

    # Load config from yaml file
    with open('AlphaZero/config/game.yaml') as f:
        game_selection = yaml.load(f)['game']
    config_path = os.path.join('AlphaZero', 'config', game_selection + '.yaml')
    if not os.path.exists(config_path):
        raise NotImplementedError("{} game config file does not exist.".format(game_selection))
    else:
        with open(config_path) as c:
            game_config = yaml.load(c)
        # Load game meta information
        # game_name = config['name']
        # game_env = importlib.import_module(config['env_path'])
        # game_converter = importlib.import_module(config['game_converter_path'])
        # state_converter = importlib.import_module(config['state_converter_path'])
    with open('AlphaZero/config/rl_sys_config.yaml') as f:
        ext_config = yaml.load(f)

    cluster = tf.train.ClusterSpec(ext_config['cluster'])

    mp.freeze_support()
    # mp.set_start_method('spawn')

    # printlog('create current net')
    # curr_net = network.network(config_file="AlphaZero/network/reinforce.yaml", cluster=cluster, job='curr')

    printlog('create pipe from opti to eval')
    opti_eval_r, opti_eval_s = Block_Pipe()
    printlog('create pipe from eval to selfplay')
    eval_selfplay_r, eval_selfplay_s = Block_Pipe()
    printlog('create data pool')
    # dgen_opti_q = mp.Queue(8)

    with optimization.DataPool(ext_config['datapool']) as selfplay_opti_q, \
            nn_eval.NNEvaluator(cluster, game_config, ext_config['chal']) as nn_eval_chal, \
            nn_eval.NNEvaluator(cluster, game_config, ext_config['best']) as nn_eval_best, \
            optimization.Optimizer(cluster, opti_eval_s, selfplay_opti_q, game_config, ext_config['optimizer']) as opti, \
            evaluator.Evaluator(nn_eval_chal, nn_eval_best, opti_eval_r, eval_selfplay_s, game_config, ext_config['evaluator']) as eval_, \
            selfplay.Selfplay(nn_eval_best, eval_selfplay_r, selfplay_opti_q, game_config, ext_config['selfplay']) as selfplay_:

        # opti.run()
        while True:
            time.sleep(30)
