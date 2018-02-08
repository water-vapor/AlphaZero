import os
import time

import numpy as np
import yaml
from tqdm import tqdm

from AlphaZero.network.main import Network

go_config_path = os.path.join('AlphaZero', 'config', 'go.yaml')
with open(go_config_path) as c:
    game_config = yaml.load(c)

num_iters = 100
batch_size = 8
test_net = Network(game_config, num_gpu=1, mode="NCHW")
w = game_config['board_width']
h = game_config['board_height']
f = game_config['history_step'] * \
    game_config['planes_per_step'] + game_config['additional_planes']
p = game_config['flat_move_output']

start = time.time()
for _ in tqdm(range(num_iters)):
    state = np.random.normal(size=(batch_size, f, w, h))
    action = np.random.normal(size=(batch_size, p))
    result = np.random.normal(size=(batch_size,))
    test_net.update((state, action, result))
end = time.time()
print("Average time of update per iter: {}".format((end - start) / num_iters))

start = time.time()
for _ in tqdm(range(num_iters)):
    state = np.random.normal(size=(batch_size, f, h, w))
    test_net.response((state,))
end = time.time()
print("Average time of response per iter: {}".format((end - start) / num_iters))
