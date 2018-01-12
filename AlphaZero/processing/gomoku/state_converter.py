import yaml
import os
import numpy as np
import AlphaZero.env.gomoku as gomoku

config_path = os.path.join('AlphaZero', 'config', 'gomoku.yaml')
with open(config_path) as c:
    config = yaml.load(c)

wid = config['board_width']
hei = config['board_height']


##
# individual feature functions (state --> tensor) begin here
##

def get_board_history(state):
    """A feature encoding WHITE and BLACK on separate planes of recent history_length states
    """

    planes = np.zeros((2 * config['history_step'], state.size, state.size))

    for t in range(config['history_step'] - 1):
        planes[2 * t, :, :] = state.board_history[t] == state.current_player  # own stone
        planes[2 * t + 1, :, :] = state.board_history[t] == -state.current_player  # opponent stone

    planes[2 * config['history_step'] - 2, :, :] = state.board == state.current_player  # own stone
    planes[2 * config['history_step'] - 1, :, :] = state.board == -state.current_player  # opponent stone
    return planes


# named features and their sizes are defined here
FEATURES = {
    "board_history": {
        "size": config['history_step'] * config['planes_per_step'],
        "function": get_board_history
    },
    "color": {
        "size": 1,
        "function": lambda state: np.ones((1, state.size, state.size)) *
                                  (state.current_player == gomoku.BLACK)
    }
}

DEFAULT_FEATURES = ["board_history", "color"]


class StateTensorConverter(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, feature_list=DEFAULT_FEATURES):
        """create a preprocessor object that will concatenate together the
        given list of features
        """

        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in FEATURES:
                self.processors[i] = FEATURES[feat]["function"]
                self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("unknown feature: %s" % feat)

    def state_to_tensor(self, state):
        """Convert a GameState to a Theano-compatible tensor
        """
        feat_tensors = [proc(state) for proc in self.processors]

        # concatenate along feature dimension then add in a singleton 'batch' dimension
        f, sz = self.output_dim, state.size

        tensor = np.concatenate(feat_tensors).reshape((1, f, sz, sz))
        tensor = tensor.astype(np.int8)

        return tensor


class TensorActionConverter(object):
    """ a class to convert output tensors from NN to action tuples

    """

    def __init__(self):
        pass

    def tensor_to_action(self, tensor):
        """

        Args:
            tensor: a 1D prob tensor with length 225

        Returns:
            a list of (action, prob)
        """
        res = [((i // wid, i % hei), tensor[i]) for i in range(wid * hei)]

        return res


def lr_reflection(action_prob):
    """ Flips the coordinate of action probability vector like np.fliplr
        Modification is made in place

    Args:
        action_prob: action probabilities

    Returns:
        None
    """
    for i in range(wid * hei):
        action_prob[i] = ((action_prob[i][0][0], wid - 1 - action_prob[i][0][1]), action_prob[i][1])


def reverse_nprot90(action_prob, transform_id):
    """ Reverse the coordinate transform of np.rot90 performed in go.Gamestate.transform
        Rotate the coordinates by Pi/4 * id clockwise

    Args:
        action_prob: action probability vector
        id: argument passed to np.rot90

    Returns:
        None
    """
    if transform_id == 0:
        return
    elif transform_id == 1:
        for i in range(wid * hei):
            action_prob[i] = ((action_prob[i][0][1], wid - 1 - action_prob[i][0][0]), action_prob[i][1])
        return
    elif transform_id == 2:
        for i in range(wid * hei):
            action_prob[i] = ((hei - 1 - action_prob[i][0][0], wid - 1 - action_prob[i][0][1]), action_prob[i][1])
        return
    elif transform_id == 3:
        for i in range(wid * hei):
            action_prob[i] = ((hei - 1 - action_prob[i][0][1], action_prob[i][0][0]), action_prob[i][1])
        return
    else:
        raise NotImplementedError


def reverse_transform(action_prob, transform_id):
    """ Reverse the coordinates for go.GameState.transform
        The function make modifications in place

    Args:
        action_prob: list of (action, prob)
        transform_id: number used to perform the transform, range: [0, 7]

    Returns:
        None
    """
    if transform_id // 4 == 1:
        lr_reflection(action_prob)
    reverse_nprot90(action_prob, transform_id % 4)
