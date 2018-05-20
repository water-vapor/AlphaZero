import multiprocessing as mp
import os
import traceback as tb

import numpy as np

from AlphaZero.train.parallel.util import ServerClientConn, printlog


class DataPool:
    """
    This class stores the training data and handles data sending and receiving.

    Args:
        ext_config: A dictionary of system configuration
    """
    def __init__(self, ext_config):
        self.data_pool = None

        self.start_training = mp.Semaphore(0)
        self.training_started = False

        self.pool_capacity = ext_config['pool_size']
        self.pool_end_index = 0
        self.is_full = False
        self.start_data_size = ext_config['start_data_size']
        self.store_path = ext_config.get('store_path')
        self.load_prev = ext_config.get('load_prev')

        conn_num = ext_config['conn_num']
        self.server_client_conn = ServerClientConn(conn_num)

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
            if not os.path.isdir(self.store_path):
                os.mkdir(self.store_path)
            for file in os.listdir(self.store_path):
                if file.endswith('.npz'):
                    loaded = np.load(os.path.join(self.store_path, file))
                    self.merge_data((loaded['arr_0'], loaded['arr_1'], loaded['arr_2']))
                    count += 1
        while True:
            (value, req_type), s_conn = self.server_client_conn.get()
            if req_type == 'put':
                # printlog('get packet')
                data = value
                self.merge_data(data)
                if self.store_path is not None:
                    np.savez_compressed(os.path.join(self.store_path, str(count)), *data)
                count += 1
                s_conn.send('done')
            elif req_type == 'get':
                # printlog('sample data')
                batch_size = value
                if not self.is_full:
                    idxs = np.random.choice(range(self.pool_end_index), batch_size)
                else:
                    idxs = np.random.choice(range(self.pool_capacity), batch_size)
                data = (self.data_pool[0][idxs], self.data_pool[1][idxs], self.data_pool[2][idxs])
                s_conn.send(data)

    def merge_data(self, data):
        if self.data_pool is None:
            printlog('init pool')
            self.data_pool = []
            for i, item in enumerate(data):
                shape = [*item.shape]
                shape[0] = self.pool_capacity
                self.data_pool.append(np.zeros(shape))
        printlog('add data to pool')
        if not self.pool_end_index + data[0].shape[0] > self.pool_capacity:
            for i, item in enumerate(data):
                self.data_pool[i][self.pool_end_index:self.pool_end_index + item.shape[0]] = item
            self.pool_end_index = self.pool_end_index + data[0].shape[0]
        else:
            self.is_full = True
            for i, item in enumerate(data):
                self.data_pool[i][self.pool_end_index:self.pool_capacity] = item[:self.pool_capacity - self.pool_end_index]
                self.data_pool[i][:item.shape[0]-self.pool_capacity+self.pool_end_index] = item[self.pool_capacity - self.pool_end_index:]
            self.pool_end_index = data[0].shape[0] - self.pool_capacity + self.pool_end_index
        if self.is_full:
            printlog('delete old data')
            printlog('data size: {}'.format(self.pool_capacity))
        else:
            printlog('data size: {}'.format(self.pool_end_index))
        if (not self.training_started) and (self.is_full or self.pool_end_index > self.start_data_size):
            self.start_training.release()
            self.training_started = True

    def put(self, data):
        self.server_client_conn.req((data, 'put'))

    def get(self, batch_size):
        data = self.server_client_conn.req((batch_size, 'get'))
        return data