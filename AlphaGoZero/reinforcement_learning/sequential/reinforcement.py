import argparse, re
import numpy as np
from AlphaGoZero.Network.main import Network
from AlphaGoZero.reinforcement_learning.sequential.optimization import optimize, sgf_to_h5
from AlphaGoZero.reinforcement_learning.sequential.evaluator import evaluate
from AlphaGoZero.reinforcement_learning.sequential.selfplay import selfplay


def get_current_time():
    return '_'.join(re.findall('\d+', str(np.datetime64('now'))))


def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning trainer')
    parser.add_argument("--model", "-m", help='Path to existing best model, starts from scratch if not specified',
                        default=None)
    parser.add_argument("--directory", "-d", help="Folder to store generated models and data", default=None)
    args = parser.parse_args()
    if args.model is None:
        random_model = Network()
        best = get_current_time()
        random_model.save(best)
    else:
        best = args.model

    best_defeated = True
    new_model = best
    # Store best models self-play data in a sequential order, since optimization requires 500,000/25,000 recent
    # self-play games
    best_models_data = []

    # The evolution logic is not clearly described in the paper.
    # Assuming the training happens sequentially, always optimize the newest model
    while True:
        if best_defeated:
            best = new_model
            selfplay(best, best + '_data')
            sgf_to_h5(best + '_data', best + '_data', 'train.h5')
            best_models_data.append(best + '_data')
        prev_model = new_model
        new_model = get_current_time()
        # Optimize with recently generated self play data, which is in a list of folders
        optimize(best_models_data[- min(20, len(best_models)):], prev_model, new_model)
        best_defeated = evaluate(best, new_model)


if __name__ == '__main__':
    main()
