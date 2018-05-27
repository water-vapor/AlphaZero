import tensorflow as tf
from AlphaZero.network.util import batch_norm, linear


class Model(object):
    """
    Neural network for AlphaGoZero. As described in "Mastering the game of Go without human knowledge".

    args:
        game_config: the rules and size of the game
        train_config: defines the size of the network and configurations in model training.
        data_format: input format, either "NCHW" or "NHWC".
    """

    def __init__(self, game_config, train_config, data_format="NHWC"):
        self._train_config = train_config
        self._data_format = data_format
        self._game_config = game_config
        self._w = game_config['board_width']
        self._h = game_config['board_height']
        self._f = game_config['history_step'] * \
            game_config['planes_per_step'] + game_config['additional_planes']

        self.x = tf.placeholder(
            tf.float32, [None, self._f, self._h, self._w], name="x")
        if game_config['output_plane'] == 1:
            self.p = tf.placeholder(
                tf.float32, [None, game_config['flat_move_output']], name="p")
        else:
            # TODO: multiple layer output needs further modification on model structure
            self.p = tf.placeholder(tf.float32, [None, game_config['output_plane'], game_config['flat_move_output']],
                                    name="p")
        self.v = tf.placeholder(tf.float32, [None], name="v")
        self.is_train = tf.placeholder(tf.bool, [], name="is_train")

        self._build_forward()
        self._build_loss()

    def _build_forward(self):
        config = self._train_config

        if self._data_format == "NHWC":
            inputs = tf.transpose(self.x, [0, 2, 3, 1])
        else:
            inputs = self.x

        W0 = tf.get_variable("W0", [3, 3, self._f, 256])
        R = tf.nn.conv2d(inputs, W0, strides=[
                         1, 1, 1, 1], padding='SAME', data_format=self._data_format)
        R = tf.nn.relu(batch_norm(R, config, self.is_train,
                                  data_format=self._data_format))

        for layer in range(config["num_blocks"]):
            with tf.variable_scope("resblock_{}".format(layer)):
                W1 = tf.get_variable("W1", [3, 3, 256, 256])
                W2 = tf.get_variable("W2", [3, 3, 256, 256])
                R1 = tf.nn.conv2d(
                    R, W1, strides=[1, 1, 1, 1], padding='SAME', data_format=self._data_format)
                R1 = tf.nn.relu(batch_norm(
                    R1, config, self.is_train, scope="B1", data_format=self._data_format))
                R2 = tf.nn.conv2d(
                    R1, W2, strides=[1, 1, 1, 1], padding='SAME', data_format=self._data_format)
                R2 = batch_norm(R2, config, self.is_train,
                                scope="B2", data_format=self._data_format)
                R = tf.nn.relu(tf.add(R, R2))

        with tf.variable_scope("policy_head"):
            W0 = tf.get_variable("W0", [1, 1, 256, 2])
            R_p = tf.nn.conv2d(
                R, W0, strides=[1, 1, 1, 1], padding='SAME', data_format=self._data_format)
            R_p = tf.reshape(tf.nn.relu(batch_norm(
                R_p, config, self.is_train, data_format=self._data_format)), [-1, self._w * self._h * 2])
            logits = linear(R_p, self._game_config['flat_move_output'], True)
            R_p = tf.nn.softmax(logits)

        with tf.variable_scope("value_head"):
            W0 = tf.get_variable("W0", [1, 1, 256, 1])
            R_v = tf.nn.conv2d(
                R, W0, strides=[1, 1, 1, 1], padding='SAME', data_format=self._data_format)
            R_v = tf.reshape(tf.nn.relu(batch_norm(
                R_v, config, self.is_train, data_format=self._data_format)), [-1, self._w * self._h])
            R_v = tf.nn.relu(linear(R_v, 256, True, scope="F1"))
            R_v = tf.nn.tanh(tf.squeeze(
                linear(R_v, 1, True, scope="F2"), [-1]))

        self.logits = logits
        self.R_p = R_p
        self.R_v = R_v

    def _build_loss(self):
        config = self._train_config
        v_loss = tf.reduce_mean(tf.squared_difference(self.R_v, self.v) / 4.)
        p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.p))
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=float(config["l2"]))
        r_loss = tf.contrib.layers.apply_regularization(
            regularizer, tf.trainable_variables())
        self.loss = p_loss + v_loss * config["MSE_scaling"] + r_loss

    def get_loss(self):
        return self.loss
