import atexit
import traceback as tb

import h5py as h5
import numpy as np
import tensorflow as tf

import AlphaZero.network.main as network
from AlphaZero.network.supervised import shuffled_hdf5_batch_generator, evaluate
from AlphaZero.train.parallel.util import *


def kill_children():
    for p in mp.active_children():
        p.terminate()


class Optimizer:
    """
    This class owns a up-to-date neural network and updates it with the training data.

    Args:
        cluster: Tensorflow cluster spec
        s_conn: Pipe to send notification to evaluator
        data_queue: Queue from which the optimizer get data
        game_config: A dictionary of game environment configuration
        ext_config: A dictionary of system configuration
    """
    def __init__(self, cluster, s_conn, data_queue, game_config, ext_config):
        printlog('create optimizer')
        self.cluster = cluster
        self.job = ext_config['job']
        self.net = None
        self.s_conn = s_conn
        self.data_queue = data_queue

        self.num_ckpt = ext_config['num_ckpt']
        self.num_log = ext_config['num_log']
        self.num_eval = ext_config['num_eval']
        self.num_steps = ext_config['num_steps']
        self.batch_size = ext_config['batch_size']
        self.num_gpu = ext_config['num_gpu']
        self.load_path = ext_config.get('load_path')
        self.log_dir = ext_config['log_dir']

        self.eval_data_path = ext_config.get('eval_data_path')
        self.train_val_test = ext_config['train_val_test']
        self.eval_batch_size = ext_config['eval_batch_size']

        atexit.register(kill_children)
        self.proc = mp.Process(target=self.run, name='optimizer')

        self.game_config = game_config
        self.tensorboard_writer = None

    def __enter__(self):
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def run(self):
        """
        The main updating process.
        """
        self.net = network.Network(self.game_config, self.num_gpu,
                                   cluster=self.cluster, job=self.job, mode='NCHW')

        start_step = 0
        if self.load_path is not None:
            self.net.load(self.load_path)
            start_step = self.net.get_global_step()

        self.tensorboard_writer = tf.summary.FileWriter(self.log_dir)
        loss_placeholder = tf.placeholder(tf.float32)
        loss_writer = tf.summary.scalar('train/loss', loss_placeholder)

        if self.eval_data_path is not None:
            dataset = h5.File(self.eval_data_path)
            n_total_data = len(dataset["states"])
            n_val_data = int(self.train_val_test[1] * n_total_data)
            n_val_data = n_val_data - (n_val_data % self.eval_batch_size)
            shuffle_indices = np.random.permutation(n_total_data)
            val_indices = shuffle_indices[0: n_val_data]

        self.data_queue.start_training.acquire()
        printlog('training loop begin')
        for step in range(start_step, self.num_steps):
            data = self.data_queue.get(self.batch_size)
            loss = self.net.update(data)
            if step % self.num_log == 0:
                printlog('update iter', step, loss)
                summ = self.net.sess.run(loss_writer, feed_dict={loss_placeholder: loss})
                self.tensorboard_writer.add_summary(summ, step)
            if self.eval_data_path is not None and step % self.num_eval == 0:
                self.eval_model(dataset, step, self.net, val_indices, self.eval_batch_size, self.log_dir)
            if (step + 1) % self.num_ckpt == 0:
                self.net.save('./' + self.game_config['name'] + '_model/ckpt')  # TODO: use proper model name
                self.s_conn.send(step + 1)

    def eval_model(self, dataset, global_step, model, val_indices, minibatch, log_dir):
        """
        Evaluate the model and record the result with tensorboard. This evaluation is different from
        the evaluation in the three main components of the RL training pipeline.

        Args:
            dataset: The dataset
            global_step: Current global step
            model: The model to be evaluate
            val_indices: The indices of validation data
            minibatch: Batch size for the evaluation
            log_dir: Directory of tensorboard log file
        """

        val_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            dataset["results"],
            val_indices,
            minibatch,
        )
        val_loss, val_accuracy, val_mse, val_sum = evaluate(
            model, val_data_generator, tag="val")
        for summ in val_sum:
            self.tensorboard_writer.add_summary(summ, global_step)
        self.tensorboard_writer.flush()


