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
with open(supervised_config_path) as fh:
    supervised_config = yaml.load(fh)
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
        mses.append(np.mean(np.square(R_v - result)))
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
        "--patience", help="Patience of learning rate decay", type=int, default=2)
    parser.add_argument(
        "--num_batches", help="Number of batches to evaluate the network", type=int, default=100)
    parser.add_argument("--train-val-test",
                        help="Fraction of data to use for training/val/test. Must sum to 1. Invalid if restarting training",
                        nargs=3, type=float, default=[0.93, .05, .02])
    parser.add_argument(
        "--log_dir", help="Directory for storing training and evaluation event file", type=str, default="log")
    parser.add_argument(
        "--save_dir", help="Directory for storing the latest model", type=str, default="model")

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
    model = Network(game_config, args.num_gpu,
                    config_file=supervised_config_path, mode="NCHW")
    writer = tf.summary.FileWriter(args.log_dir)
    total_batches = len(train_indices) // args.minibatch
    patience = 0
    best_val_loss = 1e30
    lr = float(supervised_config["learning_rate"][0])
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
            loss = model.update(batch, lr=lr)
            if global_step % args.log_iter == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                writer.flush()
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
        print("Evaluation at step {}".format(global_step))
        eval_loss, eval_accuracy, eval_mse, eval_sum = evaluate(
            model, eval_data_generator, tag="train", max_batch=args.num_batches)
        print("Train loss {}, accuracy {}, mse {}".format(
            eval_loss, eval_accuracy, eval_mse))
        val_loss, val_accuracy, val_mse, val_sum = evaluate(
            model, val_data_generator, tag="val", max_batch=args.num_batches)
        print("Dev loss {}, accuracy {}, mse {}".format(
            val_loss, val_accuracy, val_mse))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            patience = 0
            best_val_loss = val_loss
            lr /= 5
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
    test_loss, test_accuracy, test_mse, _ = evaluate(
        model, test_data_generator, tag="test")
    print("Test loss {}, accuracy {}, mse {}".format(
        test_loss, test_accuracy, test_mse))


if __name__ == '__main__':
    run_training()
