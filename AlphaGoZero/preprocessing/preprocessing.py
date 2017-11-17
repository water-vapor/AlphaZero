import numpy as np
import AlphaGoZero.go as go
import AlphaGoZero.settings as s


##
# individual feature functions (state --> tensor) begin here
##

def get_board_history(state):
    """A feature encoding WHITE and BLACK on separate planes of recent history_length states
    """

    planes = np.zeros((2 * s.history_length, state.size, state.size))

    for t in range(s.history_length - 1):
        planes[2*t, :, :] = state.board_history[t] == state.current_player  # own stone
        planes[2*t + 1, :, :] = state.board_history[t] == -state.current_player  # opponent stone

    planes[2*s.history_length - 2, :, :] = state.board == state.current_player  # own stone
    planes[2*s.history_length - 1, :, :] = state.board == -state.current_player  # opponent stone
    return planes


# named features and their sizes are defined here
FEATURES = {
    "board_history": {
        "size": 2*s.history_length,
        "function": get_board_history
    },
    "color": {
        "size": 1,
        "function": lambda state: np.ones((1, state.size, state.size)) *
        (state.current_player == go.BLACK)
    }
}

DEFAULT_FEATURES = ["board_history", "color"]


class Preprocess(object):
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
