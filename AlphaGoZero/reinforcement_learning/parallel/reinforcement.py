import AlphaGoZero.game.nn_eval as nn_eval
import AlphaGoZero.reinforcement_learning.parallel.evaluator as evaluator
import AlphaGoZero.reinforcement_learning.parallel.optimization as optimization
import AlphaGoZero.reinforcement_learning.parallel.selfplay as selfplay
import AlphaGoZero.Network.main as network
from AlphaGoZero.reinforcement_learning.parallel.util import *

import tensorflow as tf
import argparse
import multiprocessing as mp
import time

if __name__ == '__main__':
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
    # curr_net = network.Network(config_file="AlphaGoZero/Network/reinforce.yaml", cluster=cluster, job='curr')

    printlog('create pipe from opti to eval')
    opti_eval = Block_Pipe()
    printlog('create pipe from eval to dgen')
    eval_dgen = Block_Pipe()
    printlog('create data pool')
    # dgen_opti_q = mp.Queue(8) # TODO: queue size

    with optimization.Datapool(pool_size=5000, start_data_size=100) as dgen_opti_q, \
         nn_eval.NNEvaluator(cluster, 'chal', max_batch_size=32, name='chal_nn_eval') as nn_eval_chal, \
         nn_eval.NNEvaluator(cluster, 'best', max_batch_size=32, name='best_nn_eval') as nn_eval_best, \
         optimization.Optimizer(cluster, 'curr', opti_eval, dgen_opti_q) as opti, \
         evaluator.Evaluator(nn_eval_chal, nn_eval_best, opti_eval, eval_dgen) as eval_, \
         selfplay.Selfplay(nn_eval_best, eval_dgen, dgen_opti_q) as dgen:

        # opti.run()
        while True:
            time.sleep(30)
