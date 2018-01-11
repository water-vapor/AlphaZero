import tensorflow as tf
from AlphaZero.network.main import Network
from tqdm import tqdm
import numpy as np
import time
import yaml
import os

go_config_path = os.path.join('AlphaZero', 'config', 'go.yaml')
with open(go_config_path) as c:
    game_config = yaml.load(c)

num_iters = 100
test_net = Network(game_config, num_gpu=1, mode="NCHW")

start = time.time()
for _ in tqdm(range(num_iters)):
    state = np.random.normal(size=(8, 17, 19, 19))
    action = np.random.normal(size=(8, 362))
    result = np.random.normal(size=(8,))
    test_net.update((state, action, result))
end = time.time()
print("Average time of update per iter: {}".format((end - start) / num_iters))

start = time.time()
for _ in tqdm(range(num_iters)):
    state = np.random.normal(size=(8, 17, 19, 19))
    test_net.response((state,))
end = time.time()
print("Average time of response per iter: {}".format((end - start) / num_iters))
