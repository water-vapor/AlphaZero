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

go_config_path = os.path.join(os.path.dirname(
    __file__), '..', 'config', 'go.yaml')
with open(go_config_path) as c:
    game_config = yaml.load(c)

supervised_config_path = os.path.join(os.path.dirname(
    __file__), '..', "config", "supervised.yaml")
np.set_printoptions(threshold=np.nan)


def shuffled_hdf5_batch_generator(state_dataset, action_dataset, result_dataset, start_idx, end_idx, batch_size, flip=False):
    state_size = state_dataset.shape[1:]
    game_size = state_size[-1]
    state_dataset = state_dataset[start_idx: end_idx]
    action_dataset = action_dataset[start_idx: end_idx]
    result_dataset = result_dataset[start_idx: end_idx]

    while True:
        indices = list(range(end_idx - start_idx))
        random.shuffle(indices)
        states = [state_dataset[i] for i in indices]
        actions = [action_dataset[i] for i in indices]
        results = [result_dataset[i] for i in indices]

        X, Y, Z, i = [], [], [], 0
        for state, action, result in zip(states, actions, results):
            tmp = np.zeros([game_size * game_size + 1], dtype=np.float32)
            h, w = action[0], action[1]
            result = result[0]
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
            tmp[int(game_size * h + w)] = 1

            X.append(state)
            Y.append(tmp)
            Z.append(result)
            i += 1

            if i >= batch_size:
                yield np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(Z, dtype=np.float32)
                X, Y, Z, i = [], [], [], 0


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


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",
                        help="A .h5 file of training data", type=str)
    parser.add_argument("--num_gpu",
                        help="Number of GPU. Default: 4", type=int, default=4)
    parser.add_argument("--batch_size",
                        help="Size of minibatches. Default: 512", type=int, default=512)
    parser.add_argument("--num_epoch",
                        help="Number of epoches. Default: 500", type=int, default=500)
    parser.add_argument("--log_iter",
                        help="Number of steps to record training loss", type=int, default=100)
    parser.add_argument("--test_iter",
                        help="Number of steps to evaluate", type=int, default=1000)
    parser.add_argument("--num_batches",
                        help="Number of batches for evaluation", type=int, default=50)
    parser.add_argument("--train_val",
                        help="Fraction of data to use for training/val. Must sum to 1.",
                        nargs=2, type=float, default=[0.95, .05])
    parser.add_argument("--log_dir",
                        help="Directory for tf event file", type=str, default="log")
    parser.add_argument("--save_dir",
                        help="Directory for tf models", type=str, default="model")
    parser.add_argument("--resume",
                        help="Whether to start from a checkpoint", type=bool, default=False)

    args = parser.parse_args()

    dataset = h5.File(args.train_data, "r")
    n_total_data = len(dataset["states"])
    n_train_data = int(args.train_val[0] * n_total_data)
    n_val_data = n_total_data - n_train_data
    total_batches = n_train_data // args.batch_size

    print("Dataset loaded, {} samples, {} training samples, {} validaion samples".format(
        n_total_data, n_train_data, n_val_data))
    print("START TRAINING")

    model = Network(game_config=game_config, num_gpu=args.num_gpu,
                    train_config=supervised_config_path, load_pretrained=args.resume, data_format="NCHW")
    writer = tf.summary.FileWriter(args.log_dir, model.sess.graph)

    train_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        dataset["results"],
        0, n_train_data,
        args.batch_size,
        flip=True,
    )
    eval_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        dataset["results"],
        0, n_train_data,
        args.batch_size,
    )
    val_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        dataset["results"],
        n_train_data, n_total_data,
        args.batch_size,
    )

    for epoch in range(args.num_epoch):
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
