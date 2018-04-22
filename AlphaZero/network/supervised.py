import argparse
import os
import random
import h5py as h5
import numpy as np
import tensorflow as tf
import yaml
import multiprocessing as mp
from itertools import chain
from tqdm import tqdm

from AlphaZero.network.main import Network

maxsize = 200
num_process = 1
go_config_path = os.path.join('AlphaZero', 'config', 'go.yaml')
with open(go_config_path) as c:
    game_config = yaml.load(c)

supervised_config_path = os.path.join("AlphaZero", "config", "supervised.yaml")
np.set_printoptions(threshold=np.nan)


class shuffled_hdf5_batch_generator:

    def __init__(self, state_dataset, action_dataset, result_dataset, indices, batch_size, lock, flip=False, chunk_size=None, chunk_buffer=None):

        self.state_dataset = state_dataset
        self.action_dataset = action_dataset
        self.result_dataset = result_dataset
        self.indices = indices
        self.batch_size = batch_size
        self.flip = flip
        self.lock = lock
        self.data_size = len(state_dataset)
        self.chunk_size = 1 if chunk_size is None else chunk_size
        self.chunk_buffer = 1 if chunk_buffer is None else chunk_buffer

        self.state_size = state_dataset.shape[1:]
        self.game_size = self.state_size[-1]

    def __iter__(self):
        return self

    def __next__(self):
        states = []
        actions = []
        results = []
        with self.lock:
            for _ in range(self.batch_size // self.chunk_size):
                start = np.random.randint(
                    0, self.data_size - self.chunk_buffer + 1)
                end = start + self.chunk_buffer
                indices = np.random.choice(
                    list(range(end - start)), size=self.chunk_size)
                states.append(np.asarray(
                    self.state_dataset[start: end][indices]))
                actions.append(np.asarray(
                    self.action_dataset[start: end][indices]))
                results.append(np.asarray(
                    self.result_dataset[start: end][indices]))
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        results = np.concatenate(results)
        return self.transform(states, actions, results)

    def transform(self, states, actions, results):
        game_size = self.game_size
        state_size = self.state_size
        flip = self.flip
        batch = self.batch_size

        X = np.zeros([batch] + list(state_size), dtype=np.float32)
        Y = np.zeros([batch, game_size * game_size + 1], dtype=np.float32)
        Z = np.zeros([batch], dtype=np.float32)

        for i in range(batch):
            state = states[i]
            h, w = actions[i][0], actions[i][1]
            result = results[i][0]
            if flip:
                if h != game_size or w != 0:
                    flip_h = random.choice([True, False])
                    flip_w = random.choice([True, False])
                    if flip_h:
                        state = np.flip(state, axis=1)
                        h = game_size - 1 - h
                    if flip_w:
                        state = np.flip(state, axis=2)
                        w = game_size - 1 - w

            X[i] = state
            Y[i, int(game_size * h + w)] = 1
            Z[i] = result
        return X, Y, Z


def evaluate(network, data_generator, max_batch, tag="train"):
    losses = []
    accuracies = []
    mses = []
    for num, batch in enumerate(data_generator):
        state, action, result = batch
        loss, acc, mse = network.evaluate((state, action, result,))
        losses.append(loss)
        accuracies.append(acc)
        mses.append(mse)
        if num >= max_batch:
            break
    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    mse = np.mean(mses)
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(tag), simple_value=loss), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/acc".format(tag), simple_value=accuracy), ])
    mse_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/mse".format(tag), simple_value=mse), ])
    summary = [loss_sum, acc_sum, mse_sum]
    return loss, accuracy, mse, summary


def run_training(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """
    parser = argparse.ArgumentParser(
        description='Perform supervised training on a policy network.')
    parser.add_argument("--train_data",
                        help="A .h5 file of training data", type=str)
    parser.add_argument("--num_gpu",
                        help="Number of GPU. Default: 4", type=int, default=4)
    parser.add_argument("--batch_size",
                        help="Size of minibatches. Default: 512", type=int, default=512)
    parser.add_argument("--chunk_size",
                        help="Size of chunks in dataset. Default: 16", type=int, default=16)
    parser.add_argument("--chunk_buffer",
                        help="Size of chunk buffer in dataset. Default: 128", type=int, default=128)
    parser.add_argument("--num_epoch",
                        help="Number of epoches. Default: 500", type=int, default=500)
    parser.add_argument("--log_iter",
                        help="Number of steps to record training loss", type=int, default=100)
    parser.add_argument("--test_iter",
                        help="Number of steps to evaluate", type=int, default=1000)
    parser.add_argument("--num_batches",
                        help="Number of batches for evaluation", type=int, default=50)
    parser.add_argument("--train-val-test",
                        help="Fraction of data to use for training/val. Must sum to 1.",
                        nargs=2, type=float, default=[0.95, .05])
    parser.add_argument("--log_dir",
                        help="Directory for tf event file", type=str, default="log")
    parser.add_argument("--save_dir",
                        help="Directory for tf models", type=str, default="model")
    parser.add_argument("--resume",
                        help="load checkpoint", type=bool, default=False)

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    dataset = h5.File(args.train_data, "r")
    n_total_data = len(dataset["states"])
    n_train_data = int(args.train_val_test[0] * n_total_data)
    n_val_data = n_total_data - n_train_data

    print("Dataset loaded, {} samples, {} training samples, {} validaion samples".format(
        n_total_data, n_train_data, n_val_data))
    print("START TRAINING")

    shuffle_indices = np.random.permutation(n_total_data)
    train_indices = shuffle_indices[0: n_train_data]
    eval_indices = shuffle_indices[0: n_train_data]
    val_indices = shuffle_indices[n_train_data:]
    model = Network(game_config, args.num_gpu, pretrained=args.resume,
                    config_file=supervised_config_path, mode="NCHW")
    writer = tf.summary.FileWriter(args.log_dir, model.sess.graph)
    total_batches = len(train_indices) // args.batch_size

    lock = mp.Lock()

    train_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        dataset["results"],
        train_indices,
        args.batch_size,
        lock=lock,
        flip=True,
        chunk_size=args.chunk_size,
        chunk_buffer=args.chunk_buffer,
    )
    val_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        dataset["results"],
        val_indices,
        args.batch_size,
        lock=lock,
    )
    eval_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        dataset["results"],
        eval_indices,
        args.batch_size,
        lock=lock,
    )

    def fetch(it, q):
        while True:
            q.put(next(it))

    q = mp.Queue(maxsize)
    for _ in range(num_process):
        p = mp.Process(target=fetch, args=(train_data_generator, q,))
        p.daemon = True
        p.start()

    for epoch in range(args.num_epoch):
        print("Epoch {}".format(epoch))
        for _ in tqdm(range(total_batches), ascii=True):
            batch = q.get()
            global_step = model.get_global_step() + 1
            loss = model.update(batch)

            if global_step % args.log_iter == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                writer.flush()

            if global_step % args.test_iter == 0:
                _, _, _, eval_sum = evaluate(
                    model, eval_data_generator, args.num_batches, tag="train")
                _, _, _, val_sum = evaluate(
                    model, val_data_generator, args.num_batches, tag="val")
                for summ in chain(val_sum, eval_sum):
                    writer.add_summary(summ, global_step)
                writer.flush()
                model.save(os.path.join(args.save_dir, "model"))


if __name__ == '__main__':
    run_training()
