import importlib

import numpy as np


##
# individual feature functions (state --> tensor) begin here
##


class StateTensorConverter(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, config, feature_list=None):
        """create a preprocessor object that will concatenate together the
        given list of features
        """
        self._config = config
        self._game_env = importlib.import_module(config['env_path'])
        # named features and their sizes are defined here
        self._FEATURES = {
            "board_history": {
                "size": config['history_step'] * config['planes_per_step'],
                "function": self.get_board_history
            },
            "color": {
                "size": 1,
                "function": lambda state: np.ones((1, state.size, state.size)) *
                                          (state.current_player == self._game_env.BLACK)
            }
        }

        if feature_list is None:
            feature_list = ["board_history", "color"]
        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in self._FEATURES:
                self.processors[i] = self._FEATURES[feat]["function"]
                self.output_dim += self._FEATURES[feat]["size"]
            else:
                raise ValueError("unknown feature: %s" % feat)

    def get_board_history(self, state):
        """A feature encoding WHITE and BLACK on separate planes of recent history_length states
        """

        planes = np.zeros((2 * self._config['history_step'], state.size, state.size))

        for t in range(self._config['history_step'] - 1):
            planes[2 * t, :, :] = state.board_history[t] == state.current_player  # own stone
            planes[2 * t + 1, :, :] = state.board_history[t] == -state.current_player  # opponent stone

        planes[2 * self._config['history_step'] - 2, :, :] = state.board == state.current_player  # own stone
        planes[2 * self._config['history_step'] - 1, :, :] = state.board == -state.current_player  # opponent stone
        return planes

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

    def __init__(self, config):
        self._wid = config['board_width']
        self._hei = config['board_height']

    def tensor_to_action(self, tensor):
        """

        Args:
            tensor: a 1D prob tensor with length 225

        Returns:
            a list of (action, prob)
        """
        res = [((i // self._wid, i % self._hei), tensor[i]) for i in range(self._wid * self._hei)]

        return res


class ReverseTransformer(object):
    """

    """

    def __init__(self, config):
        self._config = config
        self._wid = config['board_width']
        self._hei = config['board_height']

    def lr_reflection(self, action_prob):
        """ Flips the coordinate of action probability vector like np.fliplr
            Modification is made in place

        Args:
            action_prob: action probabilities

        Returns:
            None
        """
        for i in range(self._wid * self._hei):
            action_prob[i] = ((action_prob[i][0][0], self._wid - 1 - action_prob[i][0][1]), action_prob[i][1])

    def reverse_nprot90(self, action_prob, transform_id):
        """ Reverse the coordinate transform of np.rot90 performed in go.Gamestate.transform
            Rotate the coordinates by Pi/4 * id clockwise

        Args:
            action_prob: action probability vector
            transform_id: argument passed to np.rot90

        Returns:
            None
        """
        if transform_id == 0:
            return
        elif transform_id == 1:
            for i in range(self._wid * self._hei):
                action_prob[i] = ((action_prob[i][0][1], self._wid - 1 - action_prob[i][0][0]), action_prob[i][1])
            return
        elif transform_id == 2:
            for i in range(self._wid * self._hei):
                action_prob[i] = (
                    (self._hei - 1 - action_prob[i][0][0], self._wid - 1 - action_prob[i][0][1]), action_prob[i][1])
            return
        elif transform_id == 3:
            for i in range(self._wid * self._hei):
                action_prob[i] = ((self._hei - 1 - action_prob[i][0][1], action_prob[i][0][0]), action_prob[i][1])
            return
        else:
            raise NotImplementedError

    def reverse_transform(self, action_prob, transform_id):
        """ Reverse the coordinates for go.GameState.transform
            The function make modifications in place

        Args:
            action_prob: list of (action, prob)
            transform_id: number used to perform the transform, range: [0, 7]

        Returns:
            None
        """
        if transform_id // 4 == 1:
            self.lr_reflection(action_prob)
        self.reverse_nprot90(action_prob, transform_id % 4)
