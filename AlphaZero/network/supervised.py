import argparse
import os
import random

import h5py as h5
import numpy as np
import tensorflow as tf
import yaml
from itertools import chain
from tqdm import tqdm

from AlphaZero.network.main import Network

go_config_path = os.path.join('AlphaZero', 'config', 'go.yaml')
with open(go_config_path) as c:
    game_config = yaml.load(c)

supervised_config_path = os.path.join("AlphaZero", "config", "supervised.yaml")
np.set_printoptions(threshold=np.nan)


def shuffled_hdf5_batch_generator(state_dataset, action_dataset, result_dataset,
                                  indices, batch_size, shuffle=True, flip=False):
    """A generator of shuffled batches of training data.
    """

    def convert_action(action, size=19):
        """Convert an action to a numpy array of shape (size * size + 1)
        """
        out_dim = size * size + 1
        if len(action) == out_dim:
            return np.asarray(action, dtype=np.float32)
        x, y = tuple(action)
        categorical = np.zeros(out_dim, dtype=np.float32)
        categorical[int(size * x + y)] = 1
        return categorical

    if shuffle:
        random.shuffle(indices)
    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape, dtype=np.float32)
    Ybatch = np.zeros(
        (batch_size, game_size * game_size + 1), dtype=np.float32)
    Zbatch = np.zeros(batch_size, dtype=np.float32)
    batch_idx = 0
    for data_idx in indices:
        state = np.array([plane for plane in state_dataset[data_idx]])
        h, w = action_dataset[data_idx]
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
        action = convert_action((h, w), game_size)
        result = result_dataset[data_idx]
        Xbatch[batch_idx] = state
        Ybatch[batch_idx] = action
        Zbatch[batch_idx] = result
        batch_idx += 1
        if batch_idx == batch_size:
            batch_idx = 0
            yield (Xbatch, Ybatch, Zbatch)


def evaluate(network, data_generator, tag="train", max_batch=None):
    accuracies = []
    mses = []
    for num, batch in enumerate(data_generator):
        state, action, result = batch
        loss, R_p, R_v = network.evaluate((state, action, result,))
        mask = np.argmax(R_p, axis=1) == np.argmax(action, axis=1)
        accuracies.append(np.mean(mask.astype(np.float32)))
        mses.append(np.mean(np.square(R_v - result)) / 4.)
        if max_batch is not None and num >= max_batch:
            break
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
    parser.add_argument("--train_data", "-D",
                        help="A .h5 file of training data")
    parser.add_argument(
        "--num_gpu", "-G", help="Number of GPU used for training. Default: 1", type=int, default=1)
    parser.add_argument(
        "--minibatch", "-B", help="Size of training data minibatches. Default: 64", type=int, default=64)
    parser.add_argument(
        "--epochs", "-E", help="Total number of iterations on the data. Default: 20", type=int, default=20)
    parser.add_argument(
        "--log_iter", help="Number of steps to record training loss", type=int, default=100)
    parser.add_argument(
        "--test_iter", help="Number of steps to calculate acc and mse", type=int, default=1000)
    parser.add_argument(
        "--num_batches", help="Number of batches to evaluate the network", type=int, default=150)
    parser.add_argument("--train-val-test",
                        help="Fraction of data to use for training/val/test. Must sum to 1. Invalid if restarting training",
                        nargs=3, type=float, default=[0.93, .05, .02])
    parser.add_argument(
        "--log_dir", help="Directory for storing training and evaluation event file", type=str, default="log")
    parser.add_argument(
        "--save_dir", help="Directory for storing the latest model", type=str, default="model")
    parser.add_argument(
        "--resume", "-R", help="load checkpoint", type=bool, default=False)

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    dataset = h5.File(args.train_data)
    n_total_data = len(dataset["states"])
    n_train_data = int(args.train_val_test[0] * n_total_data)
    n_train_data = n_train_data - (n_train_data % args.minibatch)
    n_val_data = int(args.train_val_test[1] * n_total_data)
    n_val_data = n_val_data - (n_val_data % args.minibatch)
    n_test_data = n_total_data - n_train_data - n_val_data

    print("Dataset loaded, {} samples, {} training samples, {} validaion samples, {} test samples".format(
        n_total_data, n_train_data, n_val_data, n_test_data))
    print("START TRAINING")

    shuffle_indices = np.random.permutation(n_total_data)
    train_indices = shuffle_indices[0: n_train_data]
    eval_indices = shuffle_indices[0: n_train_data]
    val_indices = shuffle_indices[n_train_data: n_train_data + n_val_data]
    test_indices = shuffle_indices[n_train_data + n_val_data:]
    model = Network(game_config, args.num_gpu, pretrained=args.resume,
                    config_file=supervised_config_path, mode="NCHW")
    writer = tf.summary.FileWriter(args.log_dir, model.sess.graph)
    total_batches = len(train_indices) // args.minibatch
    for epoch in range(args.epochs):
        train_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            dataset["results"],
            train_indices,
            args.minibatch,
            flip=True,
        )
        print("Epoch {}".format(epoch))
        for batch in tqdm(train_data_generator, total=total_batches, ascii=True):
            global_step = model.get_global_step() + 1
            loss = model.update(batch)

            if global_step % args.log_iter == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                writer.flush()

            if global_step % args.test_iter == 0:
                val_data_generator = shuffled_hdf5_batch_generator(
                    dataset["states"],
                    dataset["actions"],
                    dataset["results"],
                    val_indices,
                    args.minibatch,
                )
                eval_data_generator = shuffled_hdf5_batch_generator(
                    dataset["states"],
                    dataset["actions"],
                    dataset["results"],
                    eval_indices,
                    args.minibatch,
                )
                _, _, _, eval_sum = evaluate(
                    model, eval_data_generator, tag="train", max_batch=args.num_batches)
                _, _, _, val_sum = evaluate(
                    model, val_data_generator, tag="val", max_batch=args.num_batches)
                for summ in chain(val_sum, eval_sum):
                    writer.add_summary(summ, global_step)
                writer.flush()
                model.save(os.path.join(args.save_dir, "model"))

        test_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            dataset["results"],
            test_indices,
            args.minibatch,
        )
        test_loss, test_accuracy, test_mse, test_sum = evaluate(
            model, test_data_generator, tag="test")
        for summ in chain(val_sum, eval_sum):
            writer.add_summary(summ, global_step)


if __name__ == '__main__':
    run_training()
