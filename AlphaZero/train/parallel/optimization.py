import atexit
import traceback as tb
import socket
import pickle
import os

import numpy as np
import yaml
import tensorflow as tf
import h5py as h5

import AlphaZero.network.main as network
from AlphaZero.train.parallel.util import *
from AlphaZero.network.supervised import shuffled_hdf5_batch_generator, evaluate


def kill_children():
    for p in mp.active_children():
        p.terminate()


class Optimizer:
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
        self.writer = None

    def __enter__(self):
        self.proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def run(self):
        self.net = network.Network(self.game_config, self.num_gpu,
                                   cluster=self.cluster, job=self.job)
        if self.load_path is not None:
            self.net.load(self.load_path)

        self.writer = tf.summary.FileWriter(self.log_dir)
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
        for step in range(self.num_steps):
            data = self.data_queue.get(self.batch_size)
            loss = self.net.update(data)
            if step % self.num_log == 0:
                printlog('update iter', step, loss)
                summ = self.net.sess.run(loss_writer, feed_dict={loss_placeholder: loss})
                self.writer.add_summary(summ, step)
            if self.eval_data_path is not None and step % self.num_eval == 0:
                self.eval_model(dataset, step, self.net, val_indices, self.eval_batch_size, self.log_dir)
            if (step + 1) % self.num_ckpt == 0:
                self.net.save('./' + self.game_config['name'] + '_model/ckpt')  # TODO: use proper model name
                self.s_conn.send(step + 1)

    def eval_model(self, dataset, global_step, model, val_indices, minibatch, log_dir):

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
            self.writer.add_summary(summ, global_step)
        self.writer.flush()


class Datapool:
    def __init__(self, ext_config):
        self.data_pool = None
        self.start_training = mp.Semaphore(0)

        self.pool_size = ext_config['pool_size']
        self.start_data_size = ext_config['start_data_size']
        self.store_path = ext_config.get('store_path')
        self.load_prev = ext_config['load_prev']

        conn_num = ext_config['conn_num']
        self.rcpt = Reception(conn_num)

        self.server = mp.Process(target=self.serve, name='data_pool_server')

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.server.terminate()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def serve(self):
        printlog('start')
        count = 0
        if self.load_prev and self.store_path is not None:
            printlog('load previous data')
            for file in os.listdir(self.store_path):
                if file.endswith('.npz'):
                    loaded = np.load(os.path.join(self.store_path, file))
                    self.merge_data((loaded['arr_0'], loaded['arr_1'], loaded['arr_2']))
                    count += 1
        while True:
            (value, req_type), s_conn = self.rcpt.get()
            if req_type == 'put':
                # printlog('get packet')
                data = value
                self.merge_data(data)
                if self.data_pool[0].shape[0] > self.start_data_size:
                    self.start_training.release()
                if self.store_path is not None:
                    np.savez_compressed(os.path.join(self.store_path, str(count)), *data)
                count += 1
                s_conn.send('done')
            elif req_type == 'get':
                # printlog('sample data')
                batch_size = value
                idxs = np.random.choice(range(self.data_pool[0].shape[0]), batch_size)
                data = (self.data_pool[0][idxs], self.data_pool[1][idxs], self.data_pool[2][idxs])
                s_conn.send(data)

    def merge_data(self, data):
        if self.data_pool is None:
            printlog('init pool')
            self.data_pool = [item for item in data]
        else:
            printlog('add data to pool')
            self.data_pool = [np.concatenate([it, it_new], axis=0) for it, it_new in zip(self.data_pool, data)]
            if self.data_pool[0].shape[0] > self.pool_size:
                printlog('delete old data')
                self.data_pool = [it[-self.pool_size:] for it in self.data_pool]
        printlog('data size: ' + str(self.data_pool[0].shape[0]))

    def put(self, data):
        self.rcpt.req((data, 'put'))

    def get(self, batch_size):
        data = self.rcpt.req((batch_size, 'get'))
        return data
