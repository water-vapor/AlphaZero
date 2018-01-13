import argparse
import multiprocessing as mp
import os
import time

import tensorflow as tf
import yaml

import AlphaZero.game.nn_eval as nn_eval
import AlphaZero.train.parallel.evaluator as evaluator
import AlphaZero.train.parallel.optimization as optimization
import AlphaZero.train.parallel.selfplay as selfplay
from AlphaZero.train.parallel.util import *

if __name__ == '__main__':
    # Read the name of the game from cmd, load name.yaml from config folder
    parser = argparse.ArgumentParser(description='Performs reinforcement learning of AlphaZero.')
    parser.add_argument("--game", '-g', help="Name of the game, in lower case.", type=str, default="go")
    args = parser.parse_args()

    # Load config from yaml file
    config_path = os.path.join('AlphaZero', 'config', args.game + '.yaml')
    if not os.path.exists(config_path):
        raise NotImplementedError("{} game config file does not exist.".format(args.game))
    else:
        with open(config_path) as c:
            game_config = yaml.load(c)
        # Load game meta information
        # game_name = config['name']
        # game_env = importlib.import_module(config['env_path'])
        # game_converter = importlib.import_module(config['game_converter_path'])
        # state_converter = importlib.import_module(config['state_converter_path'])

    mp.freeze_support()
    # mp.set_start_method('spawn')

    cluster = tf.train.ClusterSpec({
        'curr': [
            'localhost:3333'
        ],
        'chal': [
            'localhost:3334'
        ],
        'best': [
            'localhost:3335'
        ]
    })

    # printlog('create current net')
    # curr_net = network.network(config_file="AlphaZero/network/reinforce.yaml", cluster=cluster, job='curr')

    printlog('create pipe from opti to eval')
    opti_eval = Block_Pipe()
    printlog('create pipe from eval to dgen')
    eval_dgen = Block_Pipe()
    printlog('create data pool')
    # dgen_opti_q = mp.Queue(8) # TODO: queue size

    with optimization.Datapool(pool_size=5000, start_data_size=100) as dgen_opti_q, \
            nn_eval.NNEvaluator(cluster, 'chal', game_config, max_batch_size=32, name='chal_nn_eval') as nn_eval_chal, \
            nn_eval.NNEvaluator(cluster, 'best', game_config, max_batch_size=32, name='best_nn_eval') as nn_eval_best, \
            optimization.Optimizer(cluster, 'curr', opti_eval, dgen_opti_q, game_config) as opti, \
            evaluator.Evaluator(nn_eval_chal, nn_eval_best, opti_eval, eval_dgen, game_config) as eval_, \
            selfplay.Selfplay(nn_eval_best, eval_dgen, dgen_opti_q, game_config) as dgen:

        # opti.run()
        while True:
            time.sleep(30)
