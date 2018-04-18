import tensorflow as tf
from AlphaZero.network.util import batch_norm, linear, get_variable_on_cpu

import AlphaZero.workaround.softmax_v2 as softmax_v2


class Model(object):
    def __init__(self, config, scope, game_config, reuse=False, mode="NHWC"):
        self.config = config
        self.scope = scope
        self.mode = mode
        self.reuse = reuse
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
        config = self.config
        _activation = tf.nn.relu

        if self.mode == "NHWC":
            inputs = tf.transpose(self.x, [0, 2, 3, 1])
        else:
            inputs = self.x

        W0 = get_variable_on_cpu("W0", [3, 3, self._f, 256])
        R = tf.nn.conv2d(inputs, W0, strides=[
            1, 1, 1, 1], padding='SAME', data_format=self.mode)
        R = _activation(batch_norm(R, config, self.is_train, mode=self.mode))

        for layer in range(config["num_blocks"]):
            with tf.variable_scope("resblock_{}".format(layer)):
                W1 = get_variable_on_cpu("W1", [3, 3, 256, 256])
                W2 = get_variable_on_cpu("W2", [3, 3, 256, 256])
                R1 = tf.nn.conv2d(
                    R, W1, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
                R1 = _activation(batch_norm(
                    R1, config, self.is_train, scope="B1", mode=self.mode))
                R2 = tf.nn.conv2d(
                    R1, W2, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
                R2 = batch_norm(R2, config, self.is_train,
                                scope="B2", mode=self.mode)
                R = _activation(tf.add(R, R2))

        with tf.variable_scope("policy_head"):
            W0 = get_variable_on_cpu("W0", [1, 1, 256, 2])
            R_p = tf.nn.conv2d(
                R, W0, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
            R_p = tf.reshape(_activation(batch_norm(
                R_p, config, self.is_train, mode=self.mode)), [-1, self._w * self._h * 2])
            logits = linear(R_p, self._game_config['flat_move_output'], True)
            R_p = tf.nn.softmax(logits)

        with tf.variable_scope("value_head"):
            W0 = get_variable_on_cpu("W0", [1, 1, 256, 1])
            R_v = tf.nn.conv2d(
                R, W0, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
            R_v = tf.reshape(_activation(batch_norm(
                R_v, config, self.is_train, mode=self.mode)), [-1, self._w * self._h])
            R_v = _activation(linear(R_v, 256, True, scope="F1"))
            R_v = tf.nn.tanh(tf.squeeze(
                linear(R_v, 1, True, scope="F2"), [-1]))

        self.logits = logits
        self.R_p = R_p
        self.R_v = R_v

    def _build_loss(self):
        v_loss = tf.reduce_mean(tf.squared_difference(self.R_v, self.v) / 4.)
        p_loss = tf.reduce_mean(softmax_v2.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=tf.stop_gradient(self.p)))
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=float(self.config["l2"]))
        r_loss = tf.contrib.layers.apply_regularization(
            regularizer, tf.trainable_variables())
        self.loss = p_loss + v_loss * self.config["MSE_scaling"] + r_loss

    def get_loss(self):
        return self.loss
