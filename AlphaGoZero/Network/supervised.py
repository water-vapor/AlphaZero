import argparse
import h5py as h5
import numpy as np

from preprocessing.preprocessing import Preprocess
from Network.main import Network


def one_hot_action(action, size=19):
    """Convert an (x,y) action into a size x size array of zeros with a 1 at x,y
    """
    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical


def shuffled_hdf5_batch_generator(state_dataset, action_dataset, result_dataset,
                                  indices, batch_size):
    """A generator of batches of training data for use with the fit_generator function
    of Keras. Data is accessed in the order of the given indices for shuffling.
    """
    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape)
    Ybatch = np.zeros((batch_size, game_size * game_size))
    Zbatch = np.zeros(batch_size)
    batch_idx = 0
    while True:
        for data_idx in indices:
            state = np.array([plane for plane in state_dataset[data_idx]])
            action_xy = tuple(action_dataset[data_idx])
            action = one_hot_action(action_xy, game_size)
            result = result_dataset[data_idx]
            Xbatch[batch_idx] = state
            Ybatch[batch_idx] = action.flatten()
            Zbatch[batch_idx] = result
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch, Zbatch)


def evaluate(network, data_generator, max_batch=None):
    losses = []
    accuracies = []
    mses = []
    for num, batch in enumerate(data_generator):
        state, action, value = batch
        loss, R_p, R_v = network.response([state])
        losses.append(loss)
        mask = np.argmax(R_p, axis=1) == np.argmax(action, axis=1)
        accuracies.append(np.mean(mask.astype(np.float32)))
        mses.append(np.mean(np.square(R_v - value)))
        if max_batch and num > max_batch:
            break
    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    mse = np.mean(mses)
    return loss, accuracy, mse


def run_training(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """
    parser = argparse.ArgumentParser(
        description='Perform supervised training on a policy network.')
    parser.add_argument("train_data", help="A .h5 file of training data")
    parser.add_argument(
        "num_gpu", "-G", help="Number of GPU used for training. Default: 1", type=int, default=1)
    parser.add_argument(
        "--minibatch", "-B", help="Size of training data minibatches. Default: 16", type=int, default=16)
    parser.add_argument(
        "--epochs", "-E", help="Total number of iterations on the data. Default: 10", type=int, default=10)
    parser.add_argument(
        "--checkpoint", "-C", help="Number of steps before each evaluation", type=int, default=3000)
    parser.add_argument("--epoch-length", "-l",
                        help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode",
                        default=False, action="store_true")
    parser.add_argument("--train-val-test", help="Fraction of data to use for training/val/test. Must sum to 1. Invalid if restarting training",
                        nargs=3, type=float, default=[0.93, .05, .02])

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    model_features = Preprocess.feature_list
    dataset = h5.File(args.train_data)

    # Verify that dataset's features match the model's expected features.
    if 'features' in dataset:
        dataset_features = dataset['features'][()]
        dataset_features = dataset_features.split(",")
        if len(dataset_features) != len(model_features) or \
           any(df != mf for (df, mf) in zip(dataset_features, model_features)):
            raise ValueError("Model JSON file expects features \n\t%s\n"
                             "But dataset contains \n\t%s" % ("\n\t".join(model_features),
                                                              "\n\t".join(dataset_features)))
        elif args.verbose:
            print("Verified that dataset features and model features exactly match.")
    else:
        # Cannot check each feature, but can check number of planes.
        n_dataset_planes = dataset["states"].shape[1]
        tmp_preprocess = Preprocess(model_features)
        n_model_planes = tmp_preprocess.output_dim
        if n_dataset_planes != n_model_planes:
            raise ValueError("Model JSON file expects a total of %d planes from features \n\t%s\n"
                             "But dataset contains %d planes" % (n_model_planes,
                                                                 "\n\t".join(
                                                                     model_features),
                                                                 n_dataset_planes))
        elif args.verbose:
            print("Verified agreement of number of model and dataset feature planes, but cannot "
                  "verify exact match using old dataset format.")

    n_total_data = len(dataset["states"])
    n_train_data = int(args.train_val_test[0] * n_total_data)
    n_train_data = n_train_data - (n_train_data % args.minibatch)
    n_val_data = n_total_data - n_train_data

    if args.verbose:
        print("datset loaded")
        print("\t%d total samples" % n_total_data)
        print("\t%d training samples" % n_train_data)
        print("\t%d validaion samples" % n_val_data)

    if args.verbose:
        print("STARTING TRAINING")

    model = Network(args.num_gpu)
    for epoch in range(args.epochs):
        shuffle_indices = np.random.permutation(n_total_data)
        train_indices = shuffle_indices[0:n_train_data]
        val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
        train_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            dataset["results"],
            train_indices,
            args.minibatch)
        for step, batch in enumerate(train_data_generator):
            model.update(batch)
            if (step + 1) % args.checkpoint == 0:
                if args.verbose:
                    print("Evaluation at step {}".format(step))
                val_data_generator = shuffled_hdf5_batch_generator(
                    dataset["states"],
                    dataset["actions"],
                    dataset["results"],
                    val_indices,
                    args.minibatch)
                loss, accuracy, mse = evaluate(model, val_data_generator)
                del val_data_generator
                if args.verbose:
                    print("Loss: {}, Accuracy: {}, MSE: {}".format(
                        loss, accuracy, mse))
        del train_data_generator


if __name__ == '__main__':
    run_training()
